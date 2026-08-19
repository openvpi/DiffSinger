import torch
from torch import Tensor
from torch.nn import Module


def apply_rotary_emb(freqs_cos: Tensor, freqs_sin: Tensor, t: Tensor, interleaved=True) -> Tensor:
    rot_dim = freqs_cos.shape[-1]
    t_to_rotate = t[..., :rot_dim]
    t_pass_through = t[..., rot_dim:]

    if interleaved:
        x = t_to_rotate.view(*t_to_rotate.shape[:-1], t_to_rotate.size(-1) // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        rotated_half = torch.stack((-x2, x1), dim=-1).reshape_as(t_to_rotate)
    else:
        x1, x2 = torch.split(t_to_rotate, t_to_rotate.size(-1) // 2, dim=-1)
        rotated_half = torch.cat((-x2, x1), dim=-1)

    t_rotated = (t_to_rotate * freqs_cos) + (rotated_half * freqs_sin)
    return torch.cat((t_rotated, t_pass_through), dim=-1)


class RotaryEmbedding(Module):
    def __init__(
            self,
            dim,
            theta=10000,
            max_seq_len=8192,
            interleaved: bool = True
    ):
        super().__init__()
        self.interleaved = interleaved
        self.cached_freqs_seq_len = max_seq_len
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        cos, sin = self._precompute_cache(max_seq_len)
        self.register_buffer('cached_cos', cos, persistent=False)
        self.register_buffer('cached_sin', sin, persistent=False)

    def _precompute_cache(self, seq_len: int):
        # Cache fp32 cos/sin, cast only at use — fp16/bf16 training must not
        # recompute trig on low-precision angles.
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum('i, j -> i j', seq, self.inv_freq.float())
        if self.interleaved:
            freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        else:
            freqs = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, seq_len: int):
        if seq_len > self.cached_freqs_seq_len:
            raise RuntimeError("sequence exceeds RoPE max_seq_len!")
        return self.cached_cos[0: seq_len].detach(), self.cached_sin[0: seq_len].detach()

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        device, dtype, seq_len = t.device, t.dtype, t.shape[-2]
        freqs_cos, freqs_sin = self.forward(seq_len=seq_len)
        freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dtype)
        return apply_rotary_emb(freqs_cos, freqs_sin, t, self.interleaved)
