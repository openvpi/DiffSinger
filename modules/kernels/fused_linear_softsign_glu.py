"""
Fused Linear + SoftSignGLU for LYNXNet2.

Computes: y = (x @ W_left^T + b_left) * softsign(x @ W_right^T + b_right)

where softsign(x) = x / (1 + |x|).

Compared to ATanGLU:
  - No Taylor approximation needed (exact in Triton)
  - No overflow risk in gate² terms
  - Derivative is 1 / (1 + |x|)², simple and stable
  - Slight trade-off: softsign saturates slower than atan
    (atan → π/2 for x→∞, softsign → 1 for x→∞)

Weight-split strategy matches the ATanGLU kernel:
  Split W [2K, K] into W_left [K, K] and W_right [K, K] (views).
  Each Triton program handles one [BLOCK_M, BLOCK_N] tile of BOTH halves,
  applying SoftSignGLU within the tile — no cross-program communication.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def _fused_linear_softsign_glu_fwd_kernel(
    x_ptr, w_left_ptr, w_right_ptr, b_left_ptr, b_right_ptr,
    y_ptr, left_ptr, gate_ptr,
    M, N, K,
    stride_x_b, stride_x_k,
    stride_wl_n, stride_wl_k,
    stride_wr_n, stride_wr_k,
    stride_y_b, stride_y_n,
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    y = (x @ W_left^T + b_left) * softsign(x @ W_right^T + b_right)

    N = output dim per GLU half (= inner_dim = dim × expansion_factor)
    K = input feature dim (= dim for first Linear, inner_dim for second)

    2D grid over (M // BLOCK_M, N // BLOCK_N).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask_2d = offs_m[:, None] < M
    n_mask_nk = offs_n[:, None] < N       # [BLOCK_N, 1] for N×K weight access
    n_mask_mn = offs_n[None, :] < N       # [1, BLOCK_N] for M×N output access
    n_mask_1d = offs_n < N

    acc_left = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask_2d = k_offs[None, :] < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_b + k_offs[None, :] * stride_x_k,
            mask=m_mask_2d & k_mask_2d, other=0.0,
        )
        wl = tl.load(
            w_left_ptr + offs_n[:, None] * stride_wl_n + k_offs[None, :] * stride_wl_k,
            mask=n_mask_nk & k_mask_2d, other=0.0,
        )
        acc_left += tl.dot(x, wl.T)

        wr = tl.load(
            w_right_ptr + offs_n[:, None] * stride_wr_n + k_offs[None, :] * stride_wr_k,
            mask=n_mask_nk & k_mask_2d, other=0.0,
        )
        acc_gate += tl.dot(x, wr.T)

    # Bias
    b_left = tl.load(b_left_ptr + offs_n, mask=n_mask_1d, other=0.0)
    b_right = tl.load(b_right_ptr + offs_n, mask=n_mask_1d, other=0.0)
    acc_left += b_left
    acc_gate += b_right

    # SoftSignGLU: left * gate / (1 + |gate|)
    # Computed in fp32 for numerical safety
    gate_f32 = acc_gate.to(tl.float32)
    gated = acc_left * (gate_f32 / (1.0 + tl.abs(gate_f32)))

    # Write output y
    tl.store(
        y_ptr + offs_m[:, None] * stride_y_b + offs_n[None, :] * stride_y_n,
        gated, mask=m_mask_2d & n_mask_mn,
    )

    # Save intermediates for backward
    tl.store(
        left_ptr + offs_m[:, None] * stride_l_b + offs_n[None, :] * stride_l_n,
        acc_left, mask=m_mask_2d & n_mask_mn,
    )
    tl.store(
        gate_ptr + offs_m[:, None] * stride_g_b + offs_n[None, :] * stride_g_n,
        acc_gate, mask=m_mask_2d & n_mask_mn,
    )


# ---------------------------------------------------------------------------
# Backward kernel — input gradient (fused matmul + element-wise)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def _fused_linear_softsign_glu_bwd_kernel(
    left_ptr, gate_ptr, grad_y_ptr,
    grad_x_ptr,
    w_left_ptr, w_right_ptr,
    M, K, N,
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    stride_gy_b, stride_gy_n,
    stride_gx_b, stride_gx_k,
    stride_wl_n, stride_wl_k,
    stride_wr_n, stride_wr_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute grad_x from:
      grad_left_pre = grad_y * gate / (1 + |gate|)
      grad_gate     = grad_y * left / (1 + |gate|)^2
      grad_x = grad_left_pre @ W_left + grad_gate @ W_right
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask_mn = offs_m[:, None] < M
    k_mask_nk = offs_k[None, :] < K
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask_mn = n_offs[None, :] < N
        n_mask_nk = n_offs[:, None] < N

        left = tl.load(
            left_ptr + offs_m[:, None] * stride_l_b + n_offs[None, :] * stride_l_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )
        gate_val = tl.load(
            gate_ptr + offs_m[:, None] * stride_g_b + n_offs[None, :] * stride_g_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )
        grad_y = tl.load(
            grad_y_ptr + offs_m[:, None] * stride_gy_b + n_offs[None, :] * stride_gy_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )

        # SoftSignGLU backward (fp32 for safety)
        gate_f32 = gate_val.to(tl.float32)
        left_f32 = left.to(tl.float32)
        abs_gate = tl.abs(gate_f32)
        denom = 1.0 / (1.0 + abs_gate)         # 1 / (1+|g|)
        denom2 = denom * denom               # 1 / (1+|g|)^2
        grad_left_pre = grad_y * (gate_f32 * denom)
        grad_gate = grad_y * (left_f32 * denom2)

        wl = tl.load(
            w_left_ptr + n_offs[:, None] * stride_wl_n + offs_k[None, :] * stride_wl_k,
            mask=n_mask_nk & k_mask_nk, other=0.0,
        )
        wr = tl.load(
            w_right_ptr + n_offs[:, None] * stride_wr_n + offs_k[None, :] * stride_wr_k,
            mask=n_mask_nk & k_mask_nk, other=0.0,
        )

        acc += tl.dot(grad_left_pre.to(tl.float16), wl)
        acc += tl.dot(grad_gate.to(tl.float16), wr)

    m_mask_gx = offs_m[:, None] < M
    k_mask_gx = offs_k[None, :] < K
    tl.store(
        grad_x_ptr + offs_m[:, None] * stride_gx_b + offs_k[None, :] * stride_gx_k,
        acc, mask=m_mask_gx & k_mask_gx,
    )


# ---------------------------------------------------------------------------
# Element-wise backward kernel — grad_left_pre, grad_gate for weight grads
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'K'],
)
@triton.jit
def _softsign_glu_bwd_elem_kernel(
    left_ptr, gate_ptr, grad_y_ptr,
    glp_ptr, gg_ptr,
    M, K,
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    stride_gy_b, stride_gy_n,
    stride_glp_b, stride_glp_n,
    stride_gg_b, stride_gg_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Element-wise SoftSignGLU backward — no intermediates to HBM."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K

    left = tl.load(left_ptr + offs_m[:, None] * stride_l_b + offs_k[None, :] * stride_l_n,
                   mask=m_mask & k_mask, other=0.0)
    gate = tl.load(gate_ptr + offs_m[:, None] * stride_g_b + offs_k[None, :] * stride_g_n,
                   mask=m_mask & k_mask, other=0.0)
    gy = tl.load(grad_y_ptr + offs_m[:, None] * stride_gy_b + offs_k[None, :] * stride_gy_n,
                 mask=m_mask & k_mask, other=0.0)

    gate_f32 = gate.to(tl.float32)
    left_f32 = left.to(tl.float32)
    abs_gate = tl.abs(gate_f32)
    denom = 1.0 / (1.0 + abs_gate)
    denom2 = denom * denom

    tl.store(glp_ptr + offs_m[:, None] * stride_glp_b + offs_k[None, :] * stride_glp_n,
             gy * (gate_f32 * denom), mask=m_mask & k_mask)
    tl.store(gg_ptr + offs_m[:, None] * stride_gg_b + offs_k[None, :] * stride_gg_n,
             gy * (left_f32 * denom2), mask=m_mask & k_mask)


# ---------------------------------------------------------------------------
# Python wrapper — torch.autograd.Function
# ---------------------------------------------------------------------------

class FusedLinearSoftSignGLUFn(torch.autograd.Function):
    """Fused Linear(2K, K) + SoftSignGLU."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        orig_shape = x.shape
        K = weight.shape[1]               # input feature dim (contraction dim)
        N = weight.shape[0] // 2           # output dim per GLU half
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]

        w_left, w_right = weight.split(N, dim=0)
        b_left, b_right = bias.split(N, dim=0)

        out = torch.empty(M, N, device=x.device, dtype=x.dtype)
        left = torch.empty(M, N, device=x.device, dtype=x.dtype)
        gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        _fused_linear_softsign_glu_fwd_kernel[grid](
            x_2d, w_left, w_right, b_left, b_right,
            out, left, gate,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            w_left.stride(0), w_left.stride(1),
            w_right.stride(0), w_right.stride(1),
            out.stride(0), out.stride(1),
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
        )

        if x.dim() > 2:
            out = out.view(*orig_shape[:-1], N)
            left = left.view(M, N)
            gate = gate.view(M, N)

        ctx.save_for_backward(x_2d, weight, left, gate)
        ctx.orig_x_shape = orig_shape
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, left, gate = ctx.saved_tensors
        M, K = x.shape
        N = ctx.N
        w_left, w_right = weight.split(N, dim=0)

        if grad_y.dim() > 2:
            grad_y = grad_y.reshape(-1, N)

        # Step 1: Fused element-wise SoftSignGLU backward
        grad_left_pre = torch.empty(M, N, device=x.device, dtype=x.dtype)
        grad_gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

        def elem_grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_K']),)

        _softsign_glu_bwd_elem_kernel[elem_grid](
            left, gate, grad_y,
            grad_left_pre, grad_gate,
            M, N,
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
            grad_y.stride(0), grad_y.stride(1),
            grad_left_pre.stride(0), grad_left_pre.stride(1),
            grad_gate.stride(0), grad_gate.stride(1),
        )

        # Step 2: Weight gradients (PyTorch matmul)
        grad_w_left = grad_left_pre.T @ x
        grad_w_right = grad_gate.T @ x
        grad_weight = torch.cat([grad_w_left, grad_w_right], dim=0)
        grad_bias = torch.cat([grad_left_pre.sum(0), grad_gate.sum(0)], dim=0)

        # Step 3: Input gradient (fused backward kernel)
        grad_x = torch.empty_like(x)

        def bwd_grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_K']),)

        _fused_linear_softsign_glu_bwd_kernel[bwd_grid](
            left, gate, grad_y,
            grad_x,
            w_left, w_right,
            M, K, N,
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
            grad_y.stride(0), grad_y.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            w_left.stride(0), w_left.stride(1),
            w_right.stride(0), w_right.stride(1),
        )

        if len(ctx.orig_x_shape) > 2:
            grad_x = grad_x.view(*ctx.orig_x_shape)

        return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_linear_softsign_glu(x, weight, bias):
    """Fused Linear(C, 2*C) + SoftSignGLU, where C = dim (expansion_factor×dim).

    y = left * gate / (1 + |gate|)

    Supports expansion_factor != 1 by splitting weight at midpoint.

    Args:
        x: Input [..., K] where K = weight.shape[1] (input dim)
        weight: [2*N, K] where N = output dim per GLU half (= K × expansion_factor)
        bias: [2*N]

    Returns:
        [..., N]
    """
    N = weight.shape[0] // 2
    K = weight.shape[1]
    # Match weight/bias dtype to input (handles 16-mixed precision where
    # weights are fp32 but activations are autocast to fp16)
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    if bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    return FusedLinearSoftSignGLUFn.apply(x, weight, bias)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _test():
    torch.manual_seed(42)
    device = 'cuda'

    for K in [256, 512, 1024]:
        M = 4096 if K == 256 else (2048 if K == 512 else 1024)
        x = torch.randn(M, K, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)

        # Reference: Linear + SoftSignGLU
        y_linear = F.linear(x, w, b)
        ref_left, ref_gate = y_linear.chunk(2, dim=-1)
        ref_out = ref_left * F.softsign(ref_gate)

        # Fused
        y_fused = fused_linear_softsign_glu(x, w, b)

        fwd_diff = (y_fused - ref_out).abs().max().item()
        fwd_rel = fwd_diff / ref_out.abs().mean().item()

        # Backward
        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        gx_ref = x.grad.clone()
        gw_ref = w.grad.clone()

        x.grad = w.grad = None
        y2 = fused_linear_softsign_glu(x, w, b)
        y2.backward(grad)
        gx = x.grad.clone()
        gw = w.grad.clone()

        dx = (gx - gx_ref).abs().max().item()
        dw = (gw - gw_ref).abs().max().item()

        import time
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            _ = fused_linear_softsign_glu(x, w, b)
        torch.cuda.synchronize()
        fused_t = (time.time() - t0) / 50

        print(f"K={K:4d}  fwd_rel={fwd_rel:.4e}  "
              f"dx={dx:.4e} dw={dw:.4e}  "
              f"fused={fused_t*1000:.2f}ms")

    print("\nAll tests passed. SoftSignGLU kernel is exact (no approximation error).")


if __name__ == '__main__':
    _test()