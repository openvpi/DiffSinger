from __future__ import annotations
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
from einops import rearrange, repeat


def rotate_half(x: Tensor, interleaved=True) -> Tensor:
    if not interleaved:
        # x_half1, x_half2 = x.chunk(2, dim=-1)
        # Using torch.split instead of chunk for ONNX export compatibility.
        x1, x2 = torch.split(x, x.size(-1) // 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs: Tensor, t: Tensor, interleaved=True) -> Tensor:
    rot_dim = freqs.shape[-1]
    t_to_rotate = t[..., :rot_dim]
    t_pass_through = t[..., rot_dim:]
    
    t_rotated = (t_to_rotate * freqs.cos()) + (rotate_half(t_to_rotate, interleaved) * freqs.sin())
    
    return torch.cat((t_rotated, t_pass_through), dim=-1)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        theta=10000,
        precompute_len=8192,
        cache_max_seq_len=8192,
        interleaved: bool = True
    ):
        super().__init__()
        self.interleaved = interleaved

        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._cache_max_seq_len = max(precompute_len, cache_max_seq_len)
        self._precomputed_len = precompute_len

        self.register_buffer('cached_freqs', None, persistent=True)
        self.cached_freqs_seq_len = 0
        
        if self._precomputed_len > 0:
            self._precompute_cache(self._precomputed_len)

    def _precompute_cache(self, seq_len: int):
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = einsum('i, j -> i j', seq, self.inv_freq)
        
        if self.interleaved:
            freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        else:
            freqs = torch.cat((freqs, freqs), dim=-1)

        self.cached_freqs = freqs
        self.cached_freqs_seq_len = seq_len

    def forward(self, t: Tensor, seq_len: int) -> Tensor:
        if self.cached_freqs is None or seq_len > self.cached_freqs_seq_len:
            self._precompute_cache(seq_len)
        
        return self.cached_freqs[0: seq_len].detach()

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        device, dtype, seq_len = t.device, t.dtype, t.shape[-2]
        freqs = self.forward(t, seq_len=seq_len)
        
        return apply_rotary_emb(freqs.to(device=device, dtype=dtype), t, self.interleaved)
