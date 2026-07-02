"""
Drop-in replacement for LYNXNet2Block with fused SoftSignGLU kernels.

The fused kernel replaces:
  nn.Linear(dim, inner_dim*2) + SoftSignGLU  →  one fused kernel call

Numerical accuracy:
  Exact computation — no approximation error.
  Forward/backward differences are pure fp16 rounding noise (<0.01% rel).

HBM savings:
  Per block: 4 writes/reads of [M, 2K] eliminated = 800 MB (M=50000, K=1024, fp16)
  Per 6-layer LYNXNet step: ~4.8 GB HBM traffic saved

ONNX export:
  Use `model.eval()` → falls back to original path → ONNX export works

Only supports LYNXNet2 with SoftSignGLU activation.
"""
import torch
import torch.nn as nn

from modules.kernels.fused_linear_softsign_glu import fused_linear_softsign_glu


def wrap_lynxnet2_block(block, glu_type='softsign_glu'):
    """Wrap an existing LYNXNet2Block to use fused forward.

    Keeps all weights in-place (state_dict compatible).
    Only modifies the forward pass.

    Args:
        block: LYNXNet2Block instance
        glu_type: Only 'softsign_glu' is supported.

    Returns:
        The same block with patched forward method.
    """
    assert glu_type == 'softsign_glu', \
        f"Only softsign_glu is supported in this branch, got {glu_type}"
    net = block.net  # nn.Sequential

    def fused_forward(self, x):
        residual = x

        # Original: LayerNorm → Transpose → Conv1d → Transpose
        x = net[0](x)  # LayerNorm
        x = net[1](x)  # Transpose
        x = net[2](x)  # Conv1d(depthwise)
        x = net[3](x)  # Transpose

        if self.training:
            # Fused: SoftSignGLU × 2
            x = fused_linear_softsign_glu(x, net[4].weight, net[4].bias)
            x = fused_linear_softsign_glu(x, net[6].weight, net[6].bias)
        else:
            # Original: Linear → SoftSignGLU → Linear → SoftSignGLU
            x = net[4](x)
            x = net[5](x)  # SoftSignGLU
            x = net[6](x)
            x = net[7](x)

        # Original: Linear → Dropout → +residual
        x = net[8](x)  # output projection
        x = net[9](x)  # Dropout
        return x + residual

    # Monkey-patch
    block.forward = fused_forward.__get__(block, type(block))
    return block


def patch_lynxnet2_model(model, glu_type='softsign_glu'):
    """Patch all LYNXNet2Blocks in a LYNXNet2 model.

    Args:
        model: LYNXNet2 instance
        glu_type: Only 'softsign_glu' is supported.
    """
    from modules.backbones.lynxnet2 import LYNXNet2Block
    patched = 0
    for i, layer in enumerate(model.residual_layers):
        if isinstance(layer, LYNXNet2Block):
            model.residual_layers[i] = wrap_lynxnet2_block(layer, glu_type=glu_type)
            patched += 1
    return patched


# ---------------------------------------------------------------------------
# Safe patching — handles both DDPM (denoise_fn) and ReFlow (velocity_fn),
# and checks that the backbone is actually a LYNXNet2 before patching.
# ---------------------------------------------------------------------------

def _patch_backbone_fn(backbone_fn, glu_type):
    """Patch a single backbone function/module if it's a LYNXNet2.

    Args:
        backbone_fn: The backbone module (e.g., diffusion.velocity_fn)
        glu_type: 'softsign_glu' only.

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


def patch_acoustic_model(model, glu_type='softsign_glu'):
    """Patch the LYNXNet2 backbone in a DiffSingerAcoustic.

    The backbone is at model.diffusion.denoise_fn (DDPM) or
    model.diffusion.velocity_fn (ReFlow).

    Returns:
        Number of blocks patched.
    """
    if hasattr(model, 'diffusion') and model.diffusion is not None:
        return patch_diffusion_module(model.diffusion, glu_type=glu_type)
    return 0


def patch_variance_model(model, glu_type='softsign_glu'):
    """Patch all LYNXNet2 backbones in a DiffSingerVariance.

    The variance model has separate predictors for pitch and other
    variances, each with their own backbone. Handles both DDPM and ReFlow.

    Returns:
        Number of blocks patched.
    """
    total = 0
    for predictor_attr in ['pitch_predictor', 'variance_predictor']:
        predictor = getattr(model, predictor_attr, None)
        if predictor is not None:
            total += patch_diffusion_module(predictor, glu_type=glu_type)
    return total


# ---------------------------------------------------------------------------
# Warmup — trigger Triton autotune before training starts
# ---------------------------------------------------------------------------

def warmup_fused_backbone(backbone, glu_type='softsign_glu', num_channels=1024):
    """Run one dummy forward+backward to trigger Triton autotune compilation
    for all fused kernels (fwd + bwd + elem). Call after patching, before
    the first real training step.

    Autotune results are cached on disk by Triton, so this only has an
    effect on the first run with a given kernel / shape / GPU combination.

    Args:
        backbone: LYNXNet2 model (already patched).
        glu_type: 'softsign_glu' (default, only supported option).
        num_channels: backbone width (1024 for acoustic, 512/384 for variance).
    """
    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    # spec shape: [B, n_feats, in_dims, T]
    B, T = 4, 500
    spec = torch.randn(B, backbone.n_feats, backbone.in_dims, T,
                       device=device, dtype=dtype, requires_grad=True)
    t = torch.randint(0, 1000, (B,), device=device).float()
    cond = torch.randn(B, 384, T, device=device, dtype=dtype)

    try:
        out = backbone(spec, t, cond=cond)
        out.sum().backward()
    except Exception as e:
        # Autotune failure should not crash training — Triton cache
        # can be built on the first real step instead.
        import warnings
        warnings.warn(f'Fused kernel warmup skipped ({e})')
    finally:
        for p in backbone.parameters():
            if p.grad is not None:
                p.grad = None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

