from __future__ import annotations

import math

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


class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, x):
        return torch.conv_tbc(x.contiguous(), self.weight, self.bias, self.padding)


class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding="SAME", kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size)
            )
        if self.act == 'relu':
            self.act_fn = ReLU()
        elif self.act == 'gelu':
            self.act_fn = GELU()
        elif self.act == 'swish':
            self.act_fn = SiLU()
        self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

    def forward(self, x):
        # x: T x B x C
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5

        x = self.act_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class BatchNorm1dTBC(nn.Module):
    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)  # [B, C, T]
        x = self.bn(x)  # [B, C, T]
        x = x.permute(2, 0, 1)  # [T, B, C]
        return x


class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, dropout=attention_dropout, bias=False,
            )
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act
        )

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x
