"""Sliding-window self-attention blocks used by the duration predictor.

The classic convolutional duration predictor is a stack of ``Conv1d`` layers. A
convolution kernel is a 3-D parameter, and optimizers that act on the *matrix*
structure of a parameter see only its last two dimensions, so a ``1x1`` kernel
degenerates into a scalar rescaling of the gradient while a larger one exposes
only ``kernel_size`` right-singular directions. The effective step size then
differs by an order of magnitude between layers of the very same stack.

The blocks here therefore keep every learned parameter a 2-D matrix (the
attention and feed-forward projections) or a 1-D gain (the normalization weights
and the relative bias), and use pre-norm placement so that the residual stream is
never rescaled.

Every operation is vectorized and keeps the sequence length dynamic, so the
module stays export-safe.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SlidingWindowAttention", "SlidingWindowBlock"]

# Large negative score used instead of -inf so that fully masked rows stay
# finite and produce exactly zero after the output mask is applied.
_MASKED_SCORE = -1e4

_ACTIVATIONS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


class SlidingWindowAttention(nn.Module):
    """Multi-head self-attention restricted to a local window.

    Every query attends to the ``2 * radius + 1`` items centered on itself,
    clipped at the sequence ends, plus a learned relative bias with one entry per
    window offset. The window is gathered with an integer index, so the module
    needs no per-position control flow and supports a dynamic sequence length.
    Its cost is ``O(T * (2 * radius + 1))`` instead of ``O(T ** 2)``, which
    matters for the exported graph because its sequence length is unbounded.
    """

    def __init__(self, hidden_size, num_heads=4, radius=8, dropout=0.0):
        """Initialize the module.

        Args:
            hidden_size (int): Number of channels of the input and output.
            num_heads (int): Number of attention heads; must divide ``hidden_size``.
            radius (int): Number of neighbours on each side that a position attends to.
            dropout (float): Dropout rate.
        """
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
        # Stored flat on purpose: a 1-D parameter follows the same optimizer rule
        # as the other 1-D parameters instead of being seen as a hidden matrix.
        # Entry ``k`` of every head weights the key at offset ``k - radius`` from
        # the query, so the table runs from the leftmost to the rightmost neighbor.
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

        # The neighbors of every query are gathered from a key/value padded on
        # both sides, so out-of-sequence entries are masked below rather than
        # bounds-checked. Positions outside the window are equally out of range.
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

        score = torch.matmul(query.unsqueeze(3), key_window.transpose(-1, -2)).squeeze(3) * self.scale

        bias_table = self.relative_bias.view(self.num_heads, self.window_size)
        score = score + bias_table[None, :, None, :]  # [H, W] => [B, H, T, W]

        window_mask = torch.index_select(
            F.pad(non_pad_mask, [radius, radius]), 1, window_index
        ).reshape(batch, frames, self.window_size)
        score = score.masked_fill(~window_mask[:, None, :, :], _MASKED_SCORE)
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)

        context = torch.matmul(weight.unsqueeze(3), value_window).squeeze(3)  # [B, H, T, Dh]
        context = context.transpose(1, 2).reshape(batch, frames, channels)
        return self.out_proj(context) * non_pad_mask[:, :, None]


class SlidingWindowBlock(nn.Module):
    """Pre-norm block: ``x = x + attn(norm(x))``, ``x = x + ffn(norm(x))``.

    The feed-forward part is built from two dense matrices and an activation, so
    every parameter of the block stays 2-D except the normalization gains.
    """

    def __init__(self, hidden_size, num_heads=4, radius=8, ffn_mult=4, ffn_act="gelu", dropout=0.0):
        """Initialize the module.

        Args:
            hidden_size (int): Number of channels of the input and output.
            num_heads (int): Number of attention heads of the attention part.
            radius (int): Number of neighbours on each side that a position attends to.
            ffn_mult (int): Expansion factor of the feed-forward part.
            ffn_act (str): Activation of the feed-forward part, one of ``_ACTIVATIONS``.
            dropout (float): Dropout rate.
        """
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
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, T, C).
            non_pad_mask (BoolTensor): Mask of the non-padded positions (B, T).

        Returns:
            Tensor: Output tensor (B, T, C); padded positions are zeroed.
        """
        mask = non_pad_mask[:, :, None]
        x = (x + self.dropout(self.attn(self.norm_attn(x), non_pad_mask))) * mask
        x = (x + self.dropout(self.ffn_2(_ACTIVATIONS[self.ffn_act](self.ffn_1(self.norm_ffn(x)))))) * mask
        return x
