"""Sliding-window self-attention blocks used by the duration predictor."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SlidingWindowAttention", "SlidingWindowBlock"]

# Keep fully masked rows finite before applying the output mask.
_MASKED_SCORE = -1e4

_ACTIVATIONS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


class SlidingWindowAttention(nn.Module):
    """Multi-head self-attention over a window of ``2 * radius + 1`` positions."""

    def __init__(self, hidden_size, num_heads=4, radius=8, dropout=0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Flatten the per-head bias to keep it in the 1-D optimizer group.
        self.relative_bias = nn.Parameter(torch.zeros(num_heads * self.window_size))

    def forward(self, x, non_pad_mask):
        """
        :param x: [B, T, C] input sequence
        :param non_pad_mask: [B, T] bool mask, True for real items
        :return: [B, T, C] attended sequence, zero at padded positions
        """
        batch, frames, channels = x.shape
        radius = self.radius
        qkv = self.qkv(x).view(batch, frames, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Gather padded key/value windows; window_mask excludes the padding.
        window_offset = torch.arange(self.window_size, device=x.device)[None, :]
        window_index = torch.arange(frames, device=x.device)[:, None] + window_offset
        window_index = window_index.reshape(-1)  # [T * W], row-major in (query, offset)
        head_shape = (batch, self.num_heads, frames, self.window_size, self.head_size)
        key_window = torch.index_select(
            F.pad(key, [0, 0, radius, radius]), 2, window_index
        ).reshape(head_shape)  # [B, H, T, W, Dh]
        value_window = torch.index_select(
            F.pad(value, [0, 0, radius, radius]), 2, window_index
        ).reshape(head_shape)

        b, h, t, dh = query.shape
        q = query.reshape(b * h, t, 1, dh)
        k = key_window.reshape(b * h, t, -1, dh).transpose(-1, -2)
        score = torch.matmul(q, k).reshape(b, h, t, -1) * self.scale

        bias_table = self.relative_bias.view(self.num_heads, self.window_size)
        score = score + bias_table[None, :, None, :]  # [H, W] => [B, H, T, W]

        window_mask = torch.index_select(
            F.pad(non_pad_mask, [radius, radius]), 1, window_index
        ).reshape(batch, frames, self.window_size)
        score = score.masked_fill(~window_mask[:, None, :, :], _MASKED_SCORE)
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)

        w = weight.reshape(b * h, t, 1, -1)
        v = value_window.reshape(b * h, t, -1, dh)
        context = torch.matmul(w, v).reshape(b, h, t, dh)  # [B, H, T, Dh]
        context = context.transpose(1, 2).reshape(batch, frames, channels)
        return self.out_proj(context) * non_pad_mask[:, :, None]


class SlidingWindowBlock(nn.Module):
    """Pre-norm attention and feed-forward residual block."""

    def __init__(self, hidden_size, num_heads=4, radius=8, ffn_mult=4, ffn_act="gelu", dropout=0.0):
        super().__init__()
        if ffn_act not in _ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {ffn_act!r}, expected one of {sorted(_ACTIVATIONS)}"
            )
        self.ffn_act = ffn_act
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.attn = SlidingWindowAttention(hidden_size, num_heads, radius, dropout)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn_1 = nn.Linear(hidden_size, ffn_mult * hidden_size)
        self.ffn_2 = nn.Linear(ffn_mult * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, non_pad_mask):
        """
        :param x: [B, T, C] input sequence
        :param non_pad_mask: [B, T] bool mask, True for real items
        :return: [B, T, C] output sequence, zero at padded positions
        """
        mask = non_pad_mask[:, :, None]
        x = (x + self.dropout(self.attn(self.norm_attn(x), non_pad_mask))) * mask
        x = (x + self.dropout(self.ffn_2(_ACTIVATIONS[self.ffn_act](self.ffn_1(self.norm_ffn(x)))))) * mask
        return x
