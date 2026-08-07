"""Fused Linear + SoftSignGLU for LYNXNet2.

Computes ``left * softsign(gate)`` directly from a Linear layer whose output
weights are split into left and gate halves. The forward kernel uses fp32
accumulators, and the backward path uses a Triton element-wise kernel followed
by cuBLAS matrix multiplications.
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # no triton installed (e.g. Windows without the build)
    _TRITON_AVAILABLE = False


# Minimum CUDA compute capability for Triton tl.dot (tensor cores).
# Volta (sm_70) is the floor; Turing sm_75 has fp16 tensor cores, Ampere
# sm_80 adds bf16. Anything older cannot run the fused kernel.
_MIN_CAPABILITY = (7, 0)

_FUSED_CAPABLE = None


def is_triton_available():
    """Whether the Triton Python package imported successfully."""
    return _TRITON_AVAILABLE


def _fused_capable():
    """True only if the current CUDA device can run the fused kernel.

    Caches once per process. Returns False when Triton is missing, the
    device is CPU, or the device's compute capability predates tensor cores.
    """
    global _FUSED_CAPABLE
    if _FUSED_CAPABLE is None:
        _FUSED_CAPABLE = False
        if _TRITON_AVAILABLE and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap >= _MIN_CAPABILITY:
                _FUSED_CAPABLE = True
    return _FUSED_CAPABLE


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # Small tiles
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            # Larger tiles; unsupported shared-memory sizes are pruned by Triton.
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        ],
        key=['M_BUCKET', 'N', 'K'],
    )
    @triton.jit
    def _fused_linear_softsign_glu_fwd_kernel(
        x_ptr, w_left_ptr, w_right_ptr, b_left_ptr, b_right_ptr,
        y_ptr, left_ptr, gate_ptr,
        M, N, K,
        M_BUCKET,  # next_power_of_2(M) — autotune key only, not used in body
        stride_x_b, stride_x_k,
        stride_wl_n, stride_wl_k,
        stride_wr_n, stride_wr_k,
        stride_y_b, stride_y_n,
        stride_l_b, stride_l_n,
        stride_g_b, stride_g_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """
        y = (x @ W_left^T + b_left) * softsign(x @ W_right^T + b_right)

        N = output dim per GLU half (= inner_dim = dim × expansion_factor)
        K = input feature dim (= dim for first Linear, inner_dim for second)

        2D grid over (M // BLOCK_M, N // BLOCK_N) with grouped ordering:
        programs are swizzled so that GROUP_M row-blocks share column tiles
        while they are still hot in L2 (standard Triton matmul swizzle).
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        # Grouped pid swizzle for L2 reuse
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

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

        # Computed in fp32 for numerical safety
        gate_f32 = acc_gate.to(tl.float32)
        ss_gate = gate_f32 / (1.0 + tl.abs(gate_f32))
        gated = acc_left * ss_gate

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
    # Element-wise backward kernel — grad_left_pre, grad_gate
    #
    # GEMMs run through cuBLAS; Triton handles the element-wise gradient.
    # ---------------------------------------------------------------------------

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        ],
        key=['N'],  # element-wise: tile choice is insensitive to M — never key on it
    )
    @triton.jit
    def _softsign_glu_bwd_elem_kernel(
        left_ptr, gate_ptr, grad_y_ptr,
        glp_ptr, gg_ptr,
        M, N,
        stride_l_b, stride_l_n,
        stride_g_b, stride_g_n,
        stride_gy_b, stride_gy_n,
        stride_glp_b, stride_glp_n,
        stride_gg_b, stride_gg_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Element-wise SoftSignGLU backward.

        For y = l * softsign(g):
            dy/dl = softsign(g)
            dy/dg = l / (1+|g|)^2
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        left = tl.load(left_ptr + offs_m[:, None] * stride_l_b + offs_n[None, :] * stride_l_n,
                       mask=m_mask & n_mask, other=0.0)
        gate = tl.load(gate_ptr + offs_m[:, None] * stride_g_b + offs_n[None, :] * stride_g_n,
                       mask=m_mask & n_mask, other=0.0)
        gy = tl.load(grad_y_ptr + offs_m[:, None] * stride_gy_b + offs_n[None, :] * stride_gy_n,
                     mask=m_mask & n_mask, other=0.0)

        gate_f32 = gate.to(tl.float32)
        left_f32 = left.to(tl.float32)
        denom_g = 1.0 / (1.0 + tl.abs(gate_f32))
        denom_g2 = denom_g * denom_g
        ss_gate = gate_f32 * denom_g
        grad_left_pre = gy * ss_gate
        grad_gate = gy * (left_f32 * denom_g2)

        tl.store(glp_ptr + offs_m[:, None] * stride_glp_b + offs_n[None, :] * stride_glp_n,
                 grad_left_pre, mask=m_mask & n_mask)
        tl.store(gg_ptr + offs_m[:, None] * stride_gg_b + offs_n[None, :] * stride_gg_n,
                 grad_gate, mask=m_mask & n_mask)


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
            if bias is not None:
                b_left, b_right = bias.split(N, dim=0)
            else:
                b_left = b_right = None

            out = torch.empty(M, N, device=x.device, dtype=x.dtype)
            left = torch.empty(M, N, device=x.device, dtype=x.dtype)
            gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

            _fused_linear_softsign_glu_fwd_kernel[grid](
                x_2d, w_left, w_right, b_left, b_right,
                out, left, gate,
                M, N, K,
                triton.next_power_of_2(M),  # M_BUCKET: bounds autotune re-runs under variable batch frame counts
                x_2d.stride(0), x_2d.stride(1),
                w_left.stride(0), w_left.stride(1),
                w_right.stride(0), w_right.stride(1),
                out.stride(0), out.stride(1),
                left.stride(0), left.stride(1),
                gate.stride(0), gate.stride(1),
            )

            if x.dim() != 2:
                out = out.view(*orig_shape[:-1], N)

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

            if grad_y.dim() != 2:
                grad_y = grad_y.reshape(-1, N)
            if not grad_y.is_contiguous():
                grad_y = grad_y.contiguous()

            # Step 1: Fused element-wise GLU backward (single Triton kernel,
            # grad_left_pre/grad_gate computed in registers, one HBM write each)
            grad_left_pre = torch.empty(M, N, device=x.device, dtype=x.dtype)
            grad_gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

            def elem_grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

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

            # Step 2/3: All backward GEMMs on cuBLAS (faster than a Triton GEMM
            # here, and preserves fp16/bf16/fp32 dtype without forced casts).
            # grad_weight assembled without torch.cat: write both halves into one
            # preallocated [2N, K] buffer via out= GEMMs.
            grad_weight = torch.empty(2 * N, K, device=x.device, dtype=x.dtype)
            torch.mm(grad_left_pre.T, x, out=grad_weight[:N])
            torch.mm(grad_gate.T, x, out=grad_weight[N:])
            grad_bias = torch.cat([grad_left_pre.sum(0), grad_gate.sum(0)], dim=0)

            # grad_x = grad_left_pre @ W_left + grad_gate @ W_right
            grad_x = torch.mm(grad_left_pre, w_left)
            grad_x.addmm_(grad_gate, w_right)

            if len(ctx.orig_x_shape) != 2:
                grad_x = grad_x.view(*ctx.orig_x_shape)

            return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _eager_linear_softsign_glu(x, weight, bias):
    """Unfused reference path for unsupported dtypes and devices."""
    linear = F.linear(x, weight, bias)
    left, gate = torch.split(linear, linear.shape[-1] // 2, dim=-1)
    return left * F.softsign(gate)


_FUSED_SUPPORTED_DTYPES = None
_FUSED_FALLBACK_LOG = {}


def _fused_supported_dtypes():
    """Dtypes the fused kernel can run on the current GPU.

    fp16 tl.dot: all tensor-core GPUs (Volta sm_70 and newer).
    bf16 tl.dot: Ampere (sm_80) and newer — RTX 4090 (sm_89) and
    RTX 5090 (sm_120) are fine, but Turing debug GPUs (RTX 20xx) are not.
    fp32 falls back to eager: training runs 16-mixed/bf16-mixed, and eager
    fp32 keeps full precision without a tf32 surprise inside the kernel.
    """
    global _FUSED_SUPPORTED_DTYPES
    if _FUSED_SUPPORTED_DTYPES is None:
        supported = {torch.float16}
        if torch.cuda.get_device_capability() >= (8, 0):
            supported.add(torch.bfloat16)
        _FUSED_SUPPORTED_DTYPES = supported
    return _FUSED_SUPPORTED_DTYPES


def fused_linear_softsign_glu(x, weight, bias):
    """Fused Linear(C, 2*N) + SoftSignGLU.

    Supports expansion_factor != 1 by splitting weight at its midpoint.
    Unsupported dtypes and devices use the eager implementation.

    Args:
        x: Input [..., K].
        weight: Linear weight [2*N, K].
        bias: Linear bias [2*N] or None.

    Returns:
        Output [..., N].
    """
    if not _TRITON_AVAILABLE or not _fused_capable():
        return _eager_linear_softsign_glu(x, weight, bias)
    if not x.is_cuda:
        return _eager_linear_softsign_glu(x, weight, bias)
    if bias is None:
        return _eager_linear_softsign_glu(x, weight, bias)

    fallback_key = (x.dtype, x.shape[-1])
    if x.dtype not in _fused_supported_dtypes():
        if _FUSED_FALLBACK_LOG.get(fallback_key, 0) < 1:
            _FUSED_FALLBACK_LOG[fallback_key] = 1
            import warnings
            warnings.warn(
                f'Fused SoftSignGLU: dtype {x.dtype} not supported for this GPU; '
                f'falling back to eager. (This message is shown once per (dtype, K) pair.)',
                stacklevel=2,
            )
        return _eager_linear_softsign_glu(x, weight, bias)
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
    """Numerical check against an fp32 reference."""
    import time

    torch.manual_seed(42)
    device = 'cuda'
    margin = 2.0

    def rel_err(actual, reference):
        return (actual.float() - reference).abs().max().item() / reference.abs().mean().item()

    for K in [256, 512, 1024]:
        M = 4096 if K == 256 else (2048 if K == 512 else 1024)
        x16 = torch.randn(M, K, device=device, dtype=torch.float16, requires_grad=True)
        w16 = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
        b16 = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)
        grad = torch.randn(M, K, device=device, dtype=torch.float16)

        x32 = x16.detach().float().requires_grad_(True)
        w32 = w16.detach().float().requires_grad_(True)
        b32 = b16.detach().float().requires_grad_(True)
        l32, g32 = torch.split(F.linear(x32, w32, b32), K, dim=-1)
        ref = l32 * F.softsign(g32)
        ref.backward(grad.float())

        l16, g16 = torch.split(F.linear(x16, w16, b16), K, dim=-1)
        y_eager = l16 * F.softsign(g16)
        y_eager.backward(grad)
        eager_errors = (
            rel_err(y_eager, ref),
            rel_err(x16.grad, x32.grad),
            rel_err(w16.grad, w32.grad),
            rel_err(b16.grad, b32.grad),
        )

        x16.grad = w16.grad = b16.grad = None
        y_fused = fused_linear_softsign_glu(x16, w16, b16)
        y_fused.backward(grad)
        fused_errors = (
            rel_err(y_fused, ref),
            rel_err(x16.grad, x32.grad),
            rel_err(w16.grad, w32.grad),
            rel_err(b16.grad, b32.grad),
        )

        for name, fused_error, eager_error in zip(
            ('fwd', 'grad_x', 'grad_w', 'grad_b'), fused_errors, eager_errors
        ):
            assert fused_error <= eager_error * margin + 1e-6, (
                f'K={K} {name}: fused={fused_error:.4e} vs eager={eager_error:.4e}'
            )

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            fused_linear_softsign_glu(x16, w16, b16)
        torch.cuda.synchronize()
        fused_t = (time.time() - t0) / 50

        print(
            f"K={K:4d} fwd={fused_errors[0]:.2e} dx={fused_errors[1]:.2e} "
            f"dw={fused_errors[2]:.2e} db={fused_errors[3]:.2e} "
            f"t={fused_t * 1000:.2f}ms"
        )

    print("\nAll tests passed: fused error is within fp16 rounding of the eager path.")


if __name__ == '__main__':
    _test()
