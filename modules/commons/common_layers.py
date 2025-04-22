from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx.operators
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention, ReLU, GELU, SiLU

import utils


class NormalInitEmbedding(torch.nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int | None = None,
            *args,
            **kwargs
    ):
        super().__init__(num_embeddings, embedding_dim, *args, padding_idx=padding_idx, **kwargs)
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)


class XavierUniformInitLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            *args,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0.)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, x, incremental_state=None, timestep=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(x, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    @staticmethod
    def max_positions():
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs_out = torch.max(-out_min, out_max).float()
            max_abs_gate = torch.max(-gate_min, gate_max).float()
            if max_abs_out * max_abs_gate > 1000:
                return (out.float() * gate.float()).clamp(-1000, 1000).half()
        return out * gate


class KaimingNormalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)
        
        
class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        filter_size_1 = filter_size
        if self.act == 'relu':
            self.act_fn = ReLU()
        elif self.act == 'gelu':
            self.act_fn = GELU()
        elif self.act == 'swish':
            self.act_fn = SiLU()
        elif self.act == 'swiglu':
            self.act_fn = SwiGLU()
            filter_size_1 = filter_size * 2
        else:
            raise ValueError(f'{act} is not a valid activation')
        self.ffn_1 = nn.Conv1d(hidden_size, filter_size_1, kernel_size, padding=kernel_size // 2)
        self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

    def forward(self, x):
        # x: B x T x C
        x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.kernel_size ** -0.5

        x = self.act_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False, rotary_embed=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        
        # Final linear layer after concatenation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Rotary Embeddings
        self.rotary_embed = rotary_embed
        
        # Initialization parameters
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        
    def forward(self, x, key_padding_mask=None):
        # x: (B, L, C)
        # key_padding_mask: (B, L)
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs to Q, K, V
        Q, K, V = torch.split(self.in_proj(x), self.embed_dim, dim=-1)
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # Apply RoPE
        if self.rotary_embed is not None:
            Q = self.rotary_embed.rotate_queries_or_keys(Q)
            K = self.rotary_embed.rotate_queries_or_keys(K)
            
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, L, L)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask to match attention scores shape
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 1, -np.inf)  # Masked positions are set to -inf

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to V
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (B, L, C)
        
        # Final linear projection
        output = self.out_proj(attn_output)  # (B, L, C)
        
        return output
        

class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, act='gelu', rotary_embed=None):
        super().__init__()
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        if rotary_embed is None:
            self.self_attn = MultiheadAttention(
                c, num_heads, dropout=attention_dropout, bias=False, batch_first=False
            )
            self.use_rope = False
        else:
            self.self_attn = MultiheadSelfAttentionWithRoPE(
                c, num_heads, dropout=attention_dropout, bias=False, rotary_embed=rotary_embed
            )
            self.use_rope = True
        self.layer_norm2 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, act=act
        )

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        if self.use_rope:
            x = self.self_attn(x, key_padding_mask=encoder_padding_mask)
        else:
            x = x.transpose(0, 1)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = x.transpose(0, 1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float())[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float())[..., None]
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
