"""Grouped matrix multiplication kernel with autograd support."""

import torch

# Import custom ops (triggers torch.library.custom_op registration as side effect)
from .custom_op import grouped_mm_forward, grouped_mm_swiglu_forward, grouped_mm_downproj_forward  # noqa: F401


def grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Grouped matrix multiplication with autograd support.

    Modern implementation using torch.library.custom_op registration.

    Args:
        mat_a: First input tensor, shape [..., M, K]
        mat_b: Second input tensor, shape [..., K, N]
        offs: Optional offset tensor for grouped operations
        bias: Optional bias tensor to add to result
        out_dtype: Optional output dtype

    Returns:
        Result tensor of shape [..., M, N]

    Examples:
        >>> # Batched matrix multiplication
        >>> a = torch.randn(8, 128, 256, dtype=torch.bfloat16, device="cuda")
        >>> b = torch.randn(8, 256, 512, dtype=torch.bfloat16, device="cuda")
        >>> out = grouped_mm(a, b)
        >>> out.shape
        torch.Size([8, 128, 512])

        >>> # With bias
        >>> bias = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
        >>> out = grouped_mm(a, b, bias=bias)

        >>> # Backward pass
        >>> a.requires_grad = True
        >>> b.requires_grad = True
        >>> out = grouped_mm(a, b)
        >>> out.sum().backward()
        >>> assert a.grad is not None
    """
    # Convert dtype to int (-1 represents None)
    out_dtype_int = -1 if out_dtype is None else out_dtype.as_integer()

    # Call the registered custom op
    return torch.ops.grouped_mm.mm(mat_a, mat_b, offs, bias, out_dtype_int)


def grouped_mm_swiglu(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused grouped GEMM + bias + gated-SwiGLU with autograd support.

    Single-kernel fusion of: GEMM -> bias add -> GPT-OSS gated SwiGLU activation.

    Args:
        mat_a: [total_M, K] bf16, ragged token tensor
        mat_b: [E, K, 2*I] bf16, interleaved gate/up weights
        offs: [E] int32, cumulative token offsets per expert
        bias: [E, 2*I] bf16, interleaved gate/up bias
        swiglu_alpha: Activation parameter (default 1.702)
        swiglu_limit: Clamp limit (default 7.0)

    Returns:
        (swiglu_out, aux_out): swiglu_out [total_M, I] bf16, aux_out [total_M, 2*I] bf16 GEMM+bias result.
        Backward contract: this op is fused-only and must be paired with grouped_mm_downproj so grad_aux
        can route back to grouped_mm::mm_swiglu backward. Standalone backward through swiglu_out is
        unsupported.
    """
    return torch.ops.grouped_mm.mm_swiglu(mat_a, mat_b, offs, bias, swiglu_alpha, swiglu_limit)


def grouped_mm_downproj(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None,
    aux_biased_acc: torch.Tensor,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Down-proj grouped_mm op with fused aux-gradient backward path.

    Forward is equivalent to grouped_mm(mat_a, mat_b, offs=offs, bias=bias).
    Backward can emit grad_aux directly for grouped_mm::mm_swiglu backward.
    """
    return torch.ops.grouped_mm.mm_downproj(
        mat_a,
        mat_b,
        offs,
        bias,
        aux_biased_acc,
        swiglu_alpha,
        swiglu_limit,
    )


__all__ = [
    "grouped_mm",
    "grouped_mm_downproj",
    "grouped_mm_swiglu",
]
