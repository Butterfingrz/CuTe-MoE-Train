#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>

#include "grouped_mm_kernel.cuh"

using at::Tensor;
using at::IntArrayRef;

Tensor _grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);

  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  const bool a_b_and_out_are_bf16 =
      (mat_a.dtype() == at::kBFloat16) && (mat_b.dtype() == at::kBFloat16) && (out_dtype_ == at::kBFloat16);
  TORCH_CHECK(
      a_b_and_out_are_bf16,
      "grouped_mm CUDA kernel currently supports only BFloat16 inputs/output. Got mat_a=",
      mat_a.scalar_type(),
      ", mat_b=",
      mat_b.scalar_type(),
      ", out_dtype=",
      out_dtype_);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
 // fast path, no d2h sync needed
  bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  return out;
}

// Fused GEMM + Bias + Gated-SwiGLU
// store_aux=true: write both aux_out [total_M, 2*I] and swiglu_out [total_M, I]
// store_aux=false: write only swiglu_out, skip aux D-store
void _grouped_mm_swiglu_cuda(
    const Tensor& mat_a,    // [total_M, K] bf16
    const Tensor& mat_b,    // [E, K, 2*I] bf16
    const Tensor& offs,     // [E] int32 cumsum
    const Tensor& bias,     // [E, 2*I] bf16
    const std::optional<at::Tensor>& aux_out, // [total_M, 2*I] bf16, required when store_aux=true
    Tensor& swiglu_out,     // [total_M, I] bf16, pre-allocated
    float swiglu_alpha,
    float swiglu_limit,
    bool store_aux) {
  TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 3,
      "grouped_mm_swiglu requires mat_a 2D and mat_b 3D (ragged-M path)");
  TORCH_CHECK(mat_a.dtype() == at::kBFloat16 && mat_b.dtype() == at::kBFloat16,
      "grouped_mm_swiglu requires BFloat16 inputs");
  TORCH_CHECK(bias.dtype() == at::kBFloat16, "bias must be BFloat16");
  TORCH_CHECK(offs.dtype() == at::kInt, "offs must be int32");
  if (store_aux) {
    TORCH_CHECK(aux_out.has_value(), "grouped_mm_swiglu(store_aux=True) requires aux_out tensor");
    TORCH_CHECK(aux_out->dtype() == at::kBFloat16, "aux_out must be BFloat16");
  }

  bf16bf16_grouped_mm_swiglu(mat_a, mat_b, offs, bias,
                              aux_out, swiglu_out,
                              swiglu_alpha, swiglu_limit,
                              store_aux);
}

// Fused down-proj dgrad + SwiGLU backward:
// grad_gate_up_packed = SwiGLU_bwd(grad_output @ down_w_t, aux_packed)
void _grouped_mm_swiglu_bwd_cuda(
    const Tensor& grad_output,    // [M, H] bf16
    const Tensor& down_w_t,       // [E, H, I] bf16
    const Tensor& offs,           // [E] int32
    const Tensor& aux_packed,     // [M, I] fp32 (packed from aux_out bf16 pairs)
    Tensor& grad_gate_up_packed,  // [M, I] fp32 (packed dgate/dup bf16 pairs)
    float swiglu_alpha,
    float swiglu_limit) {
  TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D [M, H]");
  TORCH_CHECK(down_w_t.dim() == 3, "down_w_t must be 3D [E, H, I]");
  TORCH_CHECK(aux_packed.dim() == 2, "aux_packed must be 2D [M, I]");
  TORCH_CHECK(grad_gate_up_packed.dim() == 2, "grad_gate_up_packed must be 2D [M, I]");
  TORCH_CHECK(offs.dim() == 1, "offs must be 1D [E]");

  TORCH_CHECK(
      grad_output.dtype() == at::kBFloat16 && down_w_t.dtype() == at::kBFloat16,
      "_grouped_mm_swiglu_bwd_cuda requires bf16 grad_output and down_w_t");
  TORCH_CHECK(aux_packed.dtype() == at::kFloat, "aux_packed must be float32");
  TORCH_CHECK(grad_gate_up_packed.dtype() == at::kFloat, "grad_gate_up_packed must be float32");
  TORCH_CHECK(offs.dtype() == at::kInt, "offs must be int32");

  TORCH_CHECK(
      grad_output.size(0) == aux_packed.size(0) &&
      aux_packed.sizes() == grad_gate_up_packed.sizes(),
      "Expected [M, I] packed tensors with matching shape");
  TORCH_CHECK(
      down_w_t.size(0) == offs.size(0),
      "down_w_t.shape[0] must match offs.size(0)");
  TORCH_CHECK(
      grad_output.size(1) == down_w_t.size(1),
      "grad_output.shape[1] must match down_w_t.shape[1] (H)");
  TORCH_CHECK(
      down_w_t.size(2) == aux_packed.size(1),
      "down_w_t.shape[2] must match aux_packed.shape[1] (I)");

  TORCH_CHECK(
      grad_output.device() == down_w_t.device() &&
      down_w_t.device() == offs.device() &&
      offs.device() == aux_packed.device() &&
      aux_packed.device() == grad_gate_up_packed.device(),
      "All tensors must be on the same device");

  bf16bf16_grouped_mm_swiglu_bwd(
      grad_output,
      down_w_t,
      offs,
      aux_packed,
      grad_gate_up_packed,
      swiglu_alpha,
      swiglu_limit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_grouped_mm_cuda", &_grouped_mm_cuda,
        "Grouped matrix multiplication (CUDA)",
        py::arg("mat_a"),
        py::arg("mat_b"),
        py::arg("offs") = py::none(),
        py::arg("bias") = py::none(),
        py::arg("out_dtype") = py::none());
  m.def("_grouped_mm_swiglu_cuda", &_grouped_mm_swiglu_cuda,
        "Fused grouped GEMM + bias + gated-SwiGLU (CUDA)",
        py::arg("mat_a"),
        py::arg("mat_b"),
        py::arg("offs"),
        py::arg("bias"),
        py::arg("aux_out") = py::none(),
        py::arg("swiglu_out"),
        py::arg("swiglu_alpha"),
        py::arg("swiglu_limit"),
        py::arg("store_aux") = true);
  m.def("_grouped_mm_swiglu_bwd_cuda", &_grouped_mm_swiglu_bwd_cuda,
        "Fused down-proj dgrad + SwiGLU backward (CUDA)",
        py::arg("grad_output"),
        py::arg("down_w_t"),
        py::arg("offs"),
        py::arg("aux_packed"),
        py::arg("grad_gate_up_packed"),
        py::arg("swiglu_alpha"),
        py::arg("swiglu_limit"));
}
