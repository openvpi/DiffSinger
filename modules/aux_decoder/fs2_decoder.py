import math

import torch
from torch import nn
import torch.nn.functional as F



def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape
class RelativeFFTBlock(nn.Module):
    """ FFT Block with Relative Multi-Head Attention """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(RelativeSelfAttention(hidden_channels, hidden_channels, n_heads,
                                    window_size=window_size, p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask=None):

        if x_mask is  not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        else:
            attn_mask = None

        for i in range(self.n_layers):
            if x_mask is not None:
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        if x_mask is not None:
            x = x * x_mask
        return x


class RelativeSelfAttention(nn.Module):
    """ Relative Multi-Head Attention """

    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
        super(RelativeSelfAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels,
                           t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + \
                self._attention_bias_proximal(t_s).to(
                    device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = torch.ones_like(
                    scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -1e4*(1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + \
                self._matmul_with_relative_values(
                    relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                                              slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [0, length-1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view(
            [batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """
        Bias for self-attention to encourage attention to close positions.
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        if x_mask is  not None:
            x = self.conv(x * x_mask)
        else:
            x = self.conv(x )

        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        if x_mask is not None:
            x=x * x_mask
        return x


class fs2_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y, x):
        x=(x - (-5)) / (0 - (-5)) * 2 - 1
        return nn.L1Loss()(y,x)


class attention_fs2_decoder(nn.Module):
    def __init__(self,encoder_hidden,out_dims,n_chans,kernel_size,dropout_rate,n_layers,n_heads,attention_ffn_kernel_size,parame):
        super().__init__()
        self.inconv=nn.Conv1d(encoder_hidden, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv = RelativeFFTBlock(hidden_channels=n_chans,filter_channels=n_chans*4, n_heads=n_heads, n_layers=n_layers, kernel_size=attention_ffn_kernel_size, p_dropout=dropout_rate)
        self.outconv=nn.Conv1d(n_chans, out_dims, kernel_size, stride=1, padding=(kernel_size - 1) // 2)



    def build_loss(self):

        return fs2_loss()

    def forward(self, x,infer,*args,**kwargs):
        x=x.transpose(1, 2)
        x=self.inconv(x)


        x=self.conv(x)
        x=self.outconv(x).transpose(1, 2)
        if infer:
            x=(x + 1) / 2 * (0 - (-5)) + (-5)
        return x
        pass
