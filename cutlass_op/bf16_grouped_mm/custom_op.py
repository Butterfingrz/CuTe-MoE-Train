"""Modern PyTorch custom op registration for grouped matrix multiplication."""

import torch

from ._ext import _grouped_mm_cuda, _grouped_mm_swiglu_cuda, _grouped_mm_swiglu_bwd_cuda


def _dtype_from_int(out_dtype: int) -> torch.dtype | None:
    return None if out_dtype == -1 else torch.dtype(out_dtype)


def _infer_grouped_mm_out_shape(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None,
) -> tuple[int, ...]:
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2

    if a_is_2d or b_is_2d:
        if offs is None:
            raise RuntimeError(
                "grouped_mm: `offs` must be provided when at least one input is 2D (ragged/grouped path)."
            )

    if a_is_2d:
        if b_is_2d:
            # Both inputs ragged => output is [group_count, M, N]
            return (offs.size(0), mat_a.size(0), mat_b.size(1))
        # A is ragged rows, B is per-group weights => output is [total_M, N]
        return (mat_a.size(0), mat_b.size(-1))

    if b_is_2d:
        # B is ragged cols => output is [M, total_N]
        return (mat_a.size(1), mat_b.size(1))

    # Regular batched GEMM => output is [B, M, N]
    return (mat_a.size(0), mat_a.size(1), mat_b.size(-1))


def _infer_grouped_mm_out_strides(
    out_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    *,
    a_is_2d: bool,
    b_is_2d: bool,
) -> tuple[int, ...]:
    """Infer output strides matching C++ padding/alignment for TMA."""
    # C++ pads the last dimension to a multiple of 16 bytes.
    alignment = 16 // torch.empty((), dtype=out_dtype).element_size()
    last_dim = len(out_shape) - 1
    size_padded = ((out_shape[last_dim] + alignment - 1) // alignment) * alignment

    if a_is_2d != b_is_2d:
        return (size_padded, 1)

    return (out_shape[1] * size_padded, size_padded, 1)


def _empty_grouped_mm_output(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None,
    out_dtype_int: int,
) -> torch.Tensor:
    """Create an empty output tensor matching the CUDA kernel's shape/stride contract."""
    out_dtype = _dtype_from_int(out_dtype_int) or mat_a.dtype
    out_shape = _infer_grouped_mm_out_shape(mat_a, mat_b, offs)
    out_strides = _infer_grouped_mm_out_strides(
        out_shape,
        out_dtype,
        a_is_2d=mat_a.dim() == 2,
        b_is_2d=mat_b.dim() == 2,
    )
    return torch.empty_strided(out_shape, out_strides, dtype=out_dtype, device=mat_a.device)


# todo fused bias in grouped_gemm
def _segment_sum_rows(grad_out_2d: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Compute per-group row-segment sum for ragged-M case (A is 2D, B is 3D).

    Uses FP32 accumulation to match PyTorch's nn.Linear autograd behavior,
    avoiding BF16 accumulation errors when summing over many tokens.
    """
    group_count = int(offs.numel())
    total_m, n = grad_out_2d.shape

    # Create FP32 accumulator (key fix: not grad_out_2d.dtype)
    grad_bias_fp32 = torch.zeros((group_count, n), dtype=torch.float32, device=grad_out_2d.device)

    # offs is cumulative end-indices; bucketize maps row -> group id.
    # Use right=True to correctly handle zero-token experts (consecutive equal offsets)
    row_ids = torch.arange(total_m, device=grad_out_2d.device, dtype=torch.int64)
    group_ids = torch.bucketize(row_ids, offs.to(torch.int64), right=True)

    # Accumulate in FP32 for precision with contiguous memory
    grad_out_fp32 = grad_out_2d.to(torch.float32).contiguous()
    grad_bias_fp32.index_add_(0, group_ids, grad_out_fp32)

    # Convert back to match input dtype for autograd consistency
    return grad_bias_fp32.to(grad_out_2d.dtype)


def _segment_sum_cols(grad_out_2d: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Compute per-group col-segment sum for ragged-N case (A is 3D, B is 2D).

    This matches the kernel's N<0 path where groups are packed along the output's
    column dimension, and bias is addressed with a per-group pointer plus a
    column index local to that group.

    Uses FP32 for intermediate sum to match PyTorch's nn.Linear autograd behavior.
    """
    group_count = int(offs.numel())
    m, total_n = grad_out_2d.shape

    # Sum in FP32 with contiguous memory for numerical stability
    col_sum = grad_out_2d.sum(dim=0, dtype=torch.float32).contiguous()  # [total_n] in FP32

    col_ids = torch.arange(total_n, device=grad_out_2d.device, dtype=torch.int64)
    offs_i64 = offs.to(torch.int64)
    group_ids = torch.bucketize(col_ids, offs_i64, right=False)  # [total_n] in [0, group_count)
    starts = torch.cat([offs_i64.new_zeros(1), offs_i64[:-1]])  # [group_count]
    rel_col = col_ids - starts[group_ids]

    # Create FP32 accumulator and ensure contiguous col_sum
    grad_bias_fp32 = torch.zeros((group_count, total_n), dtype=torch.float32, device=grad_out_2d.device)
    linear = group_ids * total_n + rel_col
    grad_bias_fp32.view(-1).index_copy_(0, linear, col_sum)

    # Convert back to match input dtype
    return grad_bias_fp32.to(grad_out_2d.dtype)


# ============================================================================
# 1. Forward Custom Op Registration
# ============================================================================


@torch.library.custom_op("grouped_mm::mm", mutates_args=())
def grouped_mm_forward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: int,
) -> torch.Tensor:
    dtype = _dtype_from_int(out_dtype)

    return _grouped_mm_cuda(mat_a, mat_b, offs, bias, dtype)


# ============================================================================
# 2. Shape Inference Registration (for meta tensors and compilation)
# ============================================================================


@grouped_mm_forward.register_fake
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None,
    _bias: torch.Tensor | None,
    out_dtype: int,
) -> torch.Tensor:
    """Shape inference for grouped_mm forward.

    This function does not execute actual computation, only infers output shape,
    dtype, and device. Used for torch.compile, torch.export, and other scenarios
    requiring static analysis.
    """
    dtype = _dtype_from_int(out_dtype) or mat_a.dtype
    out_shape = _infer_grouped_mm_out_shape(mat_a, mat_b, offs)
    out_strides = _infer_grouped_mm_out_strides(
        out_shape,
        dtype,
        a_is_2d=mat_a.dim() == 2,
        b_is_2d=mat_b.dim() == 2,
    )
    return torch.empty_strided(out_shape, out_strides, dtype=dtype, device=mat_a.device)


# ============================================================================
# 3. Setup Context Function (saves forward state for backward)
# ============================================================================


def grouped_mm_setup_context(ctx, *, inputs, output):  # noqa: ARG001
    """Setup context for backward pass.

    Args:
        ctx: Autograd context object
        inputs: Forward inputs (mat_a, mat_b, offs, bias, out_dtype)
        output: Forward output tensor (unused, part of PyTorch API)
    """
    mat_a, mat_b, offs, bias, out_dtype = inputs

    tensors_to_save: list[torch.Tensor] = [mat_a, mat_b]
    ctx.has_offs = offs is not None
    ctx.has_bias = bias is not None
    if offs is not None:
        tensors_to_save.append(offs)
    if bias is not None:
        tensors_to_save.append(bias)
    ctx.save_for_backward(*tensors_to_save)
    ctx.out_dtype = int(out_dtype)


# ============================================================================
# 4. Backward Function
# ============================================================================


def grouped_mm_backward(
    ctx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Backward pass for grouped_mm.

    Computes input gradients:
    - grad_mat_a = grad_out @ mat_b.T
    - grad_mat_b = mat_a.T @ grad_out
    - grad_bias = sum(grad_out, dim=-2)

    Args:
        ctx: Context containing saved forward state
        grad_out: Gradient w.r.t. output, shape [..., M, N]

    Returns:
        Tuple of input gradients: (grad_mat_a, grad_mat_b, None, grad_bias, None)
        None indicates non-differentiable inputs (offs, out_dtype)
    """
    saved = ctx.saved_tensors
    mat_a = saved[0]
    mat_b = saved[1]
    saved_idx = 2
    offs = saved[saved_idx] if ctx.has_offs else None
    saved_idx += 1 if ctx.has_offs else 0
    bias = saved[saved_idx] if ctx.has_bias else None

    # Handle empty gradients
    if not grad_out.numel():
        grad_mat_a = torch.zeros_like(mat_a) if ctx.needs_input_grad[0] else None
        grad_mat_b = torch.zeros_like(mat_b) if ctx.needs_input_grad[1] else None
        grad_bias = torch.zeros_like(bias) if (ctx.needs_input_grad[3] and ctx.has_bias) else None
        return grad_mat_a, grad_mat_b, None, grad_bias, None

    # Ensure grad_out is contiguous,expanded views with zero strides that the CUDA kernel cannot handle.
    grad_out = grad_out.contiguous()

    grad_mat_a = grad_mat_b = grad_bias = None

    # Compute grad_mat_a = grad_out @ mat_b.T
    # Column-major handling below is defensive code for the general grouped_mm::mm op
    # mm_swiglu and mm_downproj are MoE ragged-M specific — mat_a is always row-major
    if ctx.needs_input_grad[0]:
        # Check if mat_a is column-major
        mat_a_dim = mat_a.dim()
        mat_a_strides = mat_a.stride()
        mat_a_sizes = mat_a.size()

        is_col_major = mat_a_strides[mat_a_dim - 2] == 1 and mat_a_strides[mat_a_dim - 1] == mat_a_sizes[mat_a_dim - 2]

        if is_col_major:
            # Column-major path: _grouped_mm(mat_b, grad_out.T, offs).T
            grad_out_t = grad_out.transpose(-2, -1)
            grad_mat_a = torch.ops.grouped_mm.mm(
                mat_b,
                grad_out_t,
                offs,
                None,  # Calculate bias separately
                -1,
            ).transpose(-2, -1)
        else:
            # Row-major path: _grouped_mm(grad_out, mat_b.T, offs)
            mat_b_t = mat_b.transpose(-2, -1)
            grad_mat_a = torch.ops.grouped_mm.mm(
                grad_out,
                mat_b_t,
                offs,
                None,  # Calculate bias separately
                -1,
            )

    # Compute grad_mat_b = mat_a.T @ grad_out
    if ctx.needs_input_grad[1]:
        # Check if mat_b is column-major
        mat_b_dim = mat_b.dim()
        mat_b_strides = mat_b.stride()
        mat_b_sizes = mat_b.size()

        is_col_major = mat_b_strides[mat_b_dim - 2] == 1 and mat_b_strides[mat_b_dim - 1] == mat_b_sizes[mat_b_dim - 2]

        if is_col_major:
            # Column-major path: _grouped_mm(grad_out.T, mat_a, offs).T
            grad_out_t = grad_out.transpose(-2, -1)
            grad_mat_b = torch.ops.grouped_mm.mm(
                grad_out_t,
                mat_a,
                offs,
                None,  # Calculate bias separately
                -1,
            ).transpose(-2, -1)
        else:
            # Row-major path: _grouped_mm(mat_a.T, grad_out, offs)
            mat_a_t = mat_a.transpose(-2, -1)
            grad_mat_b = torch.ops.grouped_mm.mm(
                mat_a_t,
                grad_out,
                offs,
                None,  # Calculate bias separately
                -1,
            )
    # todo  bias backward fused in grouped_gemm
    # Compute grad_bias = sum(grad_out, dim=-2)
    if ctx.needs_input_grad[3] and ctx.has_bias:
        if grad_out.dim() == 3:
            # [group_count, M, N] -> [group_count, N]
            # Use FP32 accumulation for precision, then cast back
            grad_bias = grad_out.sum(dim=-2, dtype=torch.float32).to(grad_out.dtype)
        elif grad_out.dim() == 2:
            if offs is None:
                grad_bias = grad_out.sum(dim=0, keepdim=True)
            elif mat_a.dim() == 2 and mat_b.dim() == 3:
                grad_bias = _segment_sum_rows(grad_out, offs)
            elif mat_a.dim() == 3 and mat_b.dim() == 2:
                grad_bias = _segment_sum_cols(grad_out, offs)
            else:
                raise RuntimeError(
                    f"grouped_mm: unsupported (mat_a.dim, mat_b.dim) for bias backward: ({mat_a.dim()}, {mat_b.dim()})"
                )
        else:
            raise ValueError(f"Unexpected grad_out shape: {grad_out.shape}")

    # Return gradients for all inputs (None for non-differentiable)
    return grad_mat_a, grad_mat_b, None, grad_bias, None


# ============================================================================
# 5. Register Autograd
# ============================================================================

grouped_mm_forward.register_autograd(
    grouped_mm_backward,
    setup_context=grouped_mm_setup_context,
)


# ============================================================================
# 6. Fused GEMM + Bias + Gated-SwiGLU Custom Op
# ============================================================================


def _empty_swiglu_output(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    """Create empty swiglu output tensor [total_M, I] with TMA-aligned strides."""
    total_m = mat_a.size(0)
    n_out = mat_b.size(-1) // 2  # N_full / 2
    alignment = 16 // torch.empty((), dtype=torch.bfloat16).element_size()
    stride_padded = ((n_out + alignment - 1) // alignment) * alignment
    return torch.empty_strided(
        (total_m, n_out),
        (stride_padded, 1),
        dtype=torch.bfloat16,
        device=mat_a.device,
    )


def _as_packed_fp32(tensor_bf16: torch.Tensor, *, tensor_name: str) -> torch.Tensor:
    """View interleaved bf16 pair tensor [..., 2*I] as packed fp32 [..., I]."""
    if tensor_bf16.dtype != torch.bfloat16:
        raise RuntimeError(f"{tensor_name} must be bfloat16, got {tensor_bf16.dtype}")
    if tensor_bf16.size(-1) % 2 != 0:
        raise RuntimeError(f"{tensor_name}.shape[-1] must be even for bf16 pair packing")
    if tensor_bf16.stride(-1) != 1:
        raise RuntimeError(f"{tensor_name} must have stride(-1) == 1 for packed view")
    if tensor_bf16.stride(-2) % 2 != 0:
        raise RuntimeError(f"{tensor_name}.stride(-2) must be even for packed fp32 reinterpretation")
    return tensor_bf16.view(torch.float32)


def _has_effective_grad(grad: torch.Tensor | None) -> bool:
    """Return True only when grad is usable and not a dispatched ZeroTensor."""
    if grad is None or grad.numel() == 0:
        return False

    # ZeroTensor is metadata-only zero grad that should be skipped.
    is_zerotensor = getattr(torch, "_is_zerotensor", None)
    if is_zerotensor is not None:
        try:
            if bool(is_zerotensor(grad)):
                return False
        except Exception:  # pragma: no cover - defensive for older torch variants
            pass
    return True


@torch.library.custom_op("grouped_mm::mm_swiglu", mutates_args=())
def grouped_mm_swiglu_forward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor,
    swiglu_alpha: float,
    swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused grouped GEMM + bias + gated-SwiGLU forward.

    Args:
        mat_a: [total_M, K] bf16
        mat_b: [E, K, 2*I] bf16
        offs: [E] int32 cumsum offsets
        bias: [E, 2*I] bf16
        swiglu_alpha: SwiGLU alpha (1.702 for GPT-OSS)
        swiglu_limit: SwiGLU clamp limit (7.0 for GPT-OSS)

    Returns:
        (swiglu_out, aux_out): swiglu_out [total_M, I], aux_out [total_M, 2*I] GEMM+bias result
    """
    swiglu_out = _empty_swiglu_output(mat_a, mat_b, offs)
    aux_out = _empty_grouped_mm_output(mat_a, mat_b, offs, -1)
    _grouped_mm_swiglu_cuda(
        mat_a,
        mat_b,
        offs,
        bias,
        aux_out,
        swiglu_out,
        swiglu_alpha,
        swiglu_limit,
        True,
    )
    return swiglu_out, aux_out


@grouped_mm_swiglu_forward.register_fake
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    _bias: torch.Tensor,
    _swiglu_alpha: float,
    _swiglu_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape inference for fused swiglu forward."""
    return _empty_swiglu_output(mat_a, mat_b, offs), _empty_grouped_mm_output(mat_a, mat_b, offs, -1)


def grouped_mm_swiglu_setup_context(ctx, *, inputs, output):
    """Save forward state for swiglu backward.

    Saves aux_biased_acc (GEMM+bias output) so backward can consume it directly.
    """
    mat_a, mat_b, offs, bias, _swiglu_alpha, _swiglu_limit = inputs
    _swiglu_out, aux_biased_acc = output
    # Keep unused output grads as None (not materialized zero tensors).
    ctx.set_materialize_grads(False)
    # Bias value itself is not needed for backward; only its grad is produced.
    # Avoid saving it to reduce saved-tensor traffic (esp. under checkpointing/offload).
    ctx.save_for_backward(mat_a, mat_b, offs)
    ctx.bias_shape = bias.shape
    ctx.bias_dtype = bias.dtype


def grouped_mm_swiglu_backward(
    ctx,
    grad_swiglu: torch.Tensor | None,
    grad_aux: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    """Backward pass for fused GEMM + bias + gated-SwiGLU (fused-only contract).

    This op only supports the aux-gradient route from grouped_mm::mm_downproj.
    Standalone backward through swiglu_out is intentionally unsupported.
    """
    mat_a, mat_b, offs = ctx.saved_tensors

    has_grad_swiglu = _has_effective_grad(grad_swiglu)
    if has_grad_swiglu:
        raise RuntimeError(
            "grouped_mm::mm_swiglu backward received grad_swiglu, which is unsupported in fused-only mode. "
            "Ensure swiglu_out is consumed only by grouped_mm::mm_downproj."
        )

    if grad_aux is None:
        raise RuntimeError(
            "grouped_mm::mm_swiglu backward requires grad_aux from grouped_mm::mm_downproj (got grad_aux=None). "
            "Fused-only mode does not support standalone backward through swiglu_out."
        )

    has_grad_aux = _has_effective_grad(grad_aux)

    if not has_grad_aux:
        grad_mat_a = torch.zeros_like(mat_a) if ctx.needs_input_grad[0] else None
        grad_mat_b = torch.zeros_like(mat_b) if ctx.needs_input_grad[1] else None
        grad_bias = (
            torch.zeros(ctx.bias_shape, dtype=ctx.bias_dtype, device=mat_a.device) if ctx.needs_input_grad[3] else None
        )
        return grad_mat_a, grad_mat_b, None, grad_bias, None, None

    # zero-stride/expanded gradients are unsupported by grouped_mm kernels.
    grad_gate_up = grad_aux.contiguous()

    grad_mat_a = grad_mat_b = grad_bias = None

    # 2) GEMM backward: reuse existing grouped_mm op
    if ctx.needs_input_grad[0]:
        mat_b_t = mat_b.transpose(-2, -1)
        grad_mat_a = torch.ops.grouped_mm.mm(grad_gate_up, mat_b_t, offs, None, -1)

    if ctx.needs_input_grad[1]:
        mat_a_t = mat_a.transpose(-2, -1)
        grad_mat_b = torch.ops.grouped_mm.mm(mat_a_t, grad_gate_up, offs, None, -1)

    # 3) Bias backward: segment sum of grad_gate_up over ragged-M groups
    if ctx.needs_input_grad[3]:
        grad_bias = _segment_sum_rows(grad_gate_up, offs).to(ctx.bias_dtype)

    return grad_mat_a, grad_mat_b, None, grad_bias, None, None


grouped_mm_swiglu_forward.register_autograd(
    grouped_mm_swiglu_backward,
    setup_context=grouped_mm_swiglu_setup_context,
)


# ============================================================================
# 7. Opaque SwiGLU backward helper op (compile-safe)
# ============================================================================


@torch.library.custom_op("grouped_mm::mm_swiglu_bwd", mutates_args=())
def grouped_mm_swiglu_bwd_forward(
    grad_out: torch.Tensor,
    down_w_t: torch.Tensor,
    offs: torch.Tensor,
    aux_biased_acc: torch.Tensor,
    swiglu_alpha: float,
    swiglu_limit: float,
) -> torch.Tensor:
    """Fused down-proj dgrad + SwiGLU backward as an opaque custom op."""
    if not grad_out.numel():
        return torch.zeros_like(aux_biased_acc)

    grad_out = grad_out.contiguous()
    grad_aux = torch.empty_like(aux_biased_acc)
    _grouped_mm_swiglu_bwd_cuda(
        grad_out,
        down_w_t,
        offs,
        _as_packed_fp32(aux_biased_acc, tensor_name="aux_biased_acc"),
        _as_packed_fp32(grad_aux, tensor_name="grad_aux"),
        swiglu_alpha,
        swiglu_limit,
    )
    return grad_aux


@grouped_mm_swiglu_bwd_forward.register_fake
def _(
    _grad_out: torch.Tensor,
    _down_w_t: torch.Tensor,
    _offs: torch.Tensor,
    aux_biased_acc: torch.Tensor,
    _swiglu_alpha: float,
    _swiglu_limit: float,
) -> torch.Tensor:
    """Shape inference for grouped_mm::mm_swiglu_bwd."""
    return torch.empty_like(aux_biased_acc)


# ============================================================================
# 8. Down-proj forward op with fused aux-gradient path for SwiGLU backward
# ============================================================================


@torch.library.custom_op("grouped_mm::mm_downproj", mutates_args=())
def grouped_mm_downproj_forward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None,
    _aux_biased_acc: torch.Tensor,
    _swiglu_alpha: float,
    _swiglu_limit: float,
) -> torch.Tensor:
    """Down-proj forward that can route backward gradient through aux tensor.

    Forward is equivalent to grouped_mm::mm(mat_a, mat_b, offs, bias). In backward,
    when aux_biased_acc requires grad, this op computes grad_aux directly using the
    fused down-proj dgrad epilogue:
      grad_aux = SwiGLU_bwd(grad_out @ mat_b^T, aux_biased_acc)
    """
    if offs is None:
        raise RuntimeError("grouped_mm::mm_downproj requires offs for ragged-M path")
    if mat_a.dim() != 2 or mat_b.dim() != 3:
        raise RuntimeError("grouped_mm::mm_downproj requires mat_a 2D and mat_b 3D")

    return _grouped_mm_cuda(mat_a, mat_b, offs, bias, None)


@grouped_mm_downproj_forward.register_fake
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    _bias: torch.Tensor | None,
    _aux_biased_acc: torch.Tensor,
    _swiglu_alpha: float,
    _swiglu_limit: float,
) -> torch.Tensor:
    """Shape inference for grouped_mm::mm_downproj forward."""
    return _empty_grouped_mm_output(mat_a, mat_b, offs, -1)


def grouped_mm_downproj_setup_context(ctx, *, inputs, output):  # noqa: ARG001
    """Save forward state for down-proj backward with fused aux-grad path."""
    mat_a, mat_b, offs, bias, aux_biased_acc, swiglu_alpha, swiglu_limit = inputs
    ctx.save_for_backward(mat_a, mat_b, offs, aux_biased_acc)
    ctx.has_bias = bias is not None
    if bias is not None:
        ctx.bias_shape = bias.shape
        ctx.bias_dtype = bias.dtype
    ctx.swiglu_alpha = swiglu_alpha
    ctx.swiglu_limit = swiglu_limit


def grouped_mm_downproj_backward(
    ctx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Backward pass for grouped_mm::mm_downproj.

    Returns gradients for:
      (mat_a, mat_b, offs, bias, aux_biased_acc, swiglu_alpha, swiglu_limit)
    """
    mat_a, mat_b, offs, aux_biased_acc = ctx.saved_tensors

    if not grad_out.numel():
        grad_mat_a = (
            (None if (ctx.needs_input_grad[0] and ctx.needs_input_grad[4]) else torch.zeros_like(mat_a))
            if ctx.needs_input_grad[0]
            else None
        )
        grad_mat_b = torch.zeros_like(mat_b) if ctx.needs_input_grad[1] else None
        grad_bias = (
            torch.zeros(ctx.bias_shape, dtype=ctx.bias_dtype, device=mat_a.device)
            if (ctx.needs_input_grad[3] and ctx.has_bias)
            else None
        )
        grad_aux = torch.zeros_like(aux_biased_acc) if ctx.needs_input_grad[4] else None
        return grad_mat_a, grad_mat_b, None, grad_bias, grad_aux, None, None

    grad_out = grad_out.contiguous()

    grad_mat_a = grad_mat_b = grad_bias = grad_aux = None

    # Down-proj wgrad: grad_W_down = mat_a^T @ grad_out
    if ctx.needs_input_grad[1]:
        mat_a_t = mat_a.transpose(-2, -1)
        grad_mat_b = torch.ops.grouped_mm.mm(mat_a_t, grad_out, offs, None, -1)

    # Down-proj bias grad: segment sum over ragged rows
    if ctx.needs_input_grad[3] and ctx.has_bias:
        grad_bias = _segment_sum_rows(grad_out, offs).to(ctx.bias_dtype)

    # Fused path: produce grad_gate_up directly for grouped_mm::mm_swiglu backward.
    if ctx.needs_input_grad[4]:
        down_w_t = mat_b.transpose(-2, -1)
        grad_aux = torch.ops.grouped_mm.mm_swiglu_bwd(
            grad_out,
            down_w_t,
            offs,
            aux_biased_acc,
            ctx.swiglu_alpha,
            ctx.swiglu_limit,
        )

    # If aux-grad routing is active, skip the standard grad_mat_a path to avoid
    # an extra dgrad GEMM; grouped_mm::mm_swiglu backward consumes grad_aux directly.
    if ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]:
        mat_b_t = mat_b.transpose(-2, -1)
        grad_mat_a = torch.ops.grouped_mm.mm(grad_out, mat_b_t, offs, None, -1)

    return grad_mat_a, grad_mat_b, None, grad_bias, grad_aux, None, None


grouped_mm_downproj_forward.register_autograd(
    grouped_mm_downproj_backward,
    setup_context=grouped_mm_downproj_setup_context,
)
