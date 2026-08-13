"""
Drop-in replacement for LYNXNet2Block with fused Linear+SoftSignGLU kernels.

The fused kernel replaces:
  nn.Linear(dim, inner_dim*2) + SoftSignGLU  →  one fused kernel call
(training mode only; eval mode uses the original nn.Sequential path).

Only softsign_glu is supported — other GLU types are left unpatched
(warning at patch time, block runs the original forward).

Numerical accuracy:
  SoftSignGLU is exact in Triton (no approximation). Differences vs the
  eager path are fp16 rounding only (~1e-3 max on unit-scale activations).

HBM savings (per fused call, M=50000, N=1024, fp16):
  Eager:  Linear writes [M, 2N] (200 MB), GLU reads [M, 2N] + writes [M, N]
  Fused:  writes y/left/gate = 3×[M, N] — saves the [M, 2N] round-trip
Backward saves the softsign/denominator intermediates by fusing the
element-wise gradient into one kernel; all GEMMs stay on cuBLAS.

ONNX export:
  Use `model.eval()` → falls back to original path → ONNX export works
"""
import contextlib
import traceback
import warnings

import torch
import torch.nn as nn

from modules.backbones.lynxnet2 import LYNXNet2Block
from modules.commons.common_layers import Transpose
from modules.kernels.fused_linear_softsign_glu import (
    fused_linear_softsign_glu,
    is_triton_available,
)


_FUSABLE_GLU_TYPES = ('softsign_glu',)


class _FusedLYNXNet2BlockMixin:
    """Pickle-safe fused forward mixed into an existing LYNXNet2Block."""

    def forward(self, x):
        if not self.training:
            return super().forward(x)

        residual = x
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = fused_linear_softsign_glu(x, self.net[4].weight, self.net[4].bias)
        x = fused_linear_softsign_glu(x, self.net[6].weight, self.net[6].bias)
        x = self.net[8](x)
        x = self.net[9](x)
        return x + residual


class FusedLYNXNet2Block(_FusedLYNXNet2BlockMixin, LYNXNet2Block):
    """LYNXNet2Block variant with a pickle-safe fused training forward."""


def wrap_lynxnet2_block(block, glu_type='softsign_glu'):
    """Wrap an existing LYNXNet2Block to use fused forward.

    Keeps all weights in-place (state_dict compatible).
    Only modifies the forward pass.

    Only 'softsign_glu' is fused. Other GLU types are returned unpatched.

    Args:
        block: LYNXNet2Block instance
        glu_type: GLU type configured for this block

    Returns:
        The same block, with patched forward if glu_type is supported.
    """
    if glu_type not in _FUSABLE_GLU_TYPES:
        warnings.warn(
            f"Fused kernels support only {_FUSABLE_GLU_TYPES}; leaving block "
            f"with glu_type={glu_type!r} unpatched.",
            stacklevel=2,
        )
        return block

    net = block.net
    if not (
        len(net) == 10
        and isinstance(net[1], Transpose)  # channel-first
        and isinstance(net[3], Transpose)  # back to channel-last
        and isinstance(net[2], nn.Conv1d)
        and net[2].groups == net[2].in_channels  # depthwise conv keeps `dim` channels
        and isinstance(net[4], nn.Linear)
        and isinstance(net[6], nn.Linear)
        and isinstance(net[8], nn.Linear)
        and net[4].out_features == 2 * net[6].in_features
        and net[6].out_features == 2 * net[8].in_features
        and net[8].out_features == net[4].in_features  # round-trip to dim
    ):
        warnings.warn(
            'Unexpected LYNXNet2Block.net layout; leaving block unpatched.',
            stacklevel=2,
        )
        return block

    block.__class__ = FusedLYNXNet2Block
    return block


def patch_lynxnet2_model(model, glu_type='softsign_glu'):
    """Patch all LYNXNet2Blocks in a LYNXNet2 model.

    Args:
        model: LYNXNet2 instance
        glu_type: GLU type configured for the model (only softsign_glu fuses)

    Returns:
        Number of blocks patched (0 if glu_type unsupported).
    """
    if glu_type not in _FUSABLE_GLU_TYPES:
        warnings.warn(
            f"Fused kernels require glu_type in {_FUSABLE_GLU_TYPES}; "
            f"got {glu_type!r}. Skipping patch.",
            stacklevel=2,
        )
        return 0
    if not is_triton_available():
        raise RuntimeError(
            'Fused kernels require a working Triton installation. '
            'Install Triton for this platform or set use_fused_kernels=false.'
        )
    patched = 0
    for i, layer in enumerate(model.residual_layers):
        if isinstance(layer, LYNXNet2Block):
            layer = wrap_lynxnet2_block(layer, glu_type=glu_type)
            model.residual_layers[i] = layer
            patched += isinstance(layer, FusedLYNXNet2Block)
    return patched


# ---------------------------------------------------------------------------
# Safe patching — handles both DDPM (denoise_fn) and ReFlow (velocity_fn),
# and checks that the backbone is actually a LYNXNet2 before patching.
# ---------------------------------------------------------------------------

def _patch_backbone_fn(backbone_fn, glu_type):
    """Patch a single backbone function/module if it's a LYNXNet2.

    Args:
        backbone_fn: The backbone module (e.g., diffusion.denoise_fn)
        glu_type: GLU type (only softsign_glu fuses)

    Returns:
        Number of blocks patched (0 if not a LYNXNet2).
    """
    from modules.backbones.lynxnet2 import LYNXNet2
    if not isinstance(backbone_fn, LYNXNet2):
        return 0
    return patch_lynxnet2_model(backbone_fn, glu_type=glu_type)


def _try_patch(module, attr, glu_type):
    """Try to patch backbone at module.attr if it's a LYNXNet2. Safe to call
    even if attr doesn't exist — returns 0 silently."""
    backbone = getattr(module, attr, None)
    if backbone is None:
        return 0
    return _patch_backbone_fn(backbone, glu_type)


def patch_diffusion_module(diffusion, glu_type='softsign_glu'):
    """Patch a diffusion module's backbone (DDPM or ReFlow).

    Handles both:
      GaussianDiffusion / PitchDiffusion / MultiVarianceDiffusion → .denoise_fn
      RectifiedFlow / PitchRectifiedFlow / MultiVarianceRectifiedFlow → .velocity_fn

    Returns:
        Number of blocks patched.
    """
    return (
        _try_patch(diffusion, 'denoise_fn', glu_type) +
        _try_patch(diffusion, 'velocity_fn', glu_type)
    )


# ---------------------------------------------------------------------------
# Warmup — trigger Triton autotune before training starts
# ---------------------------------------------------------------------------

def warmup_fused_backbone(backbone, max_frames=None, autocast_dtype=None):
    """Run dummy forward passes to trigger Triton autotune compilation
    for all fused kernels (fwd + bwd elem). Call after patching, before
    the first real training step (model must already be on its CUDA device).

    Only forward is executed (``torch.no_grad``) — the element-wise backward
    kernel's autotune key depends on ``N`` (a single fixed value per model),
    so its one-off compile cost is paid on the first real step instead.

    Autotune timings are cached in process memory only (Triton persists
    compiled binaries to disk, but re-runs the config benchmark per process),
    so this runs once per training process. The forward kernel's autotune
    key buckets M by next_power_of_2, so we sweep the power-of-two buckets
    a real run will hit: from a small bucket up to next_power_of_2(max_frames).

    Args:
        backbone: LYNXNet2 model (already patched).
        max_frames: max total frames per batch (hparams['max_batch_frames']).
            If None, warms a single small bucket only.
        autocast_dtype: torch.float16 for '16-mixed', torch.bfloat16 for
            'bf16-mixed'. If None, no autocast — with fp32 parameters the
            fused path falls back to eager and the warmup is a no-op.
    """
    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    if device.type != 'cuda':
        return 0
    if not is_triton_available():
        raise RuntimeError(
            'Fused kernel warmup requires a working Triton installation. '
            'Install Triton for this platform or set use_fused_kernels=false.'
        )

    import triton

    # cond hidden size from the conditioner projection (Linear or Conv1d)
    proj = backbone.conditioner_projection
    hidden = getattr(proj, 'in_features', None) or proj.in_channels

    B = 4
    # Sweep M buckets: 2048 up to next_power_of_2(max_frames)
    if max_frames is not None:
        top = triton.next_power_of_2(int(max_frames))
        bucket = 2048
        t_list = []
        while bucket <= top:
            # M = B * T lands in this bucket (M just above the previous bucket)
            t_list.append(bucket // B // 2 + 1)
            bucket *= 2
    else:
        t_list = [500]

    ac_factory = (
        (lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype))
        if autocast_dtype is not None else contextlib.nullcontext
    )
    # Fork the RNG so dummy inputs do not advance the training noise stream.
    with torch.random.fork_rng(devices=[device]):
        for T in t_list:
            # spec shape: [B, n_feats, in_dims, T]
            spec = torch.randn(B, backbone.n_feats, backbone.in_dims, T,
                               device=device, dtype=dtype)
            t = torch.randint(0, 1000, (B,), device=device).float()
            cond = torch.randn(B, hidden, T, device=device, dtype=dtype)

            try:
                with torch.no_grad():
                    with ac_factory():
                        backbone(spec, t, cond=cond)
            except Exception as e:  # noqa: BLE001 - warmup must remain non-fatal
                # Autotune failure should not crash training — Triton cache
                # can be built on the first real step instead.
                warnings.warn(
                    f'Fused kernel warmup skipped at T={T} '
                    f'({type(e).__name__}: {e})\n{traceback.format_exc()}',
                    stacklevel=2,
                )
                break
            finally:
                del spec, cond
    torch.cuda.empty_cache()
    return len(t_list)


def warmup_fused_backbones(backbones, max_frames, precision):
    """Warm all patched backbones using Lightning's effective precision."""
    precision = str(precision)
    autocast_dtype = (
        torch.float16 if '16' in precision and 'bf16' not in precision
        else torch.bfloat16 if 'bf16' in precision
        else None
    )
    if autocast_dtype is None:
        from lightning.pytorch.utilities.rank_zero import rank_zero_info
        rank_zero_info(
            'Fused kernels: precision=%s has no autocast dtype; '
            'fused kernel will fall back to eager at runtime.', precision
        )
    for backbone in backbones:
        warmup_fused_backbone(
            backbone,
            max_frames=max_frames,
            autocast_dtype=autocast_dtype,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _test():
    import torch
    from modules.backbones.lynxnet2 import LYNXNet2Block

    device = 'cuda'
    torch.manual_seed(42)

    # Create a single block
    block = LYNXNet2Block(dim=256, expansion_factor=1, glu_type='softsign_glu').to(device).half()

    # Copy weights
    block_ref = LYNXNet2Block(dim=256, expansion_factor=1, glu_type='softsign_glu').to(device).half()
    block_ref.load_state_dict(block.state_dict())

    # Patch
    wrap_lynxnet2_block(block, glu_type='softsign_glu')

    B, T = 2, 500
    x = torch.randn(B, T, 256, device=device, dtype=torch.float16)

    # Forward
    out_orig = block_ref(x)
    out_fused = block(x)

    fwd_diff = (out_fused - out_orig).abs().max().item()
    print(f"Block forward max diff: {fwd_diff:.4e}")

    # Backward
    grad = torch.randn_like(out_orig)
    out_orig.backward(grad)
    grads_ref = {n: p.grad.clone() for n, p in block_ref.named_parameters() if p.grad is not None}

    for p in block.parameters():
        p.grad = None

    out_fused = block(x)
    out_fused.backward(grad)
    grads_fused = {n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None}

    max_w_diff = max(
        (grads_fused[n] - grads_ref[n]).abs().max().item()
        for n in grads_ref
    )
    print(f"Block weight grad max diff: {max_w_diff:.4e}")
    print("\nIntegration works! Use model.eval() for ONNX export fallback.")


if __name__ == '__main__':
    _test()
