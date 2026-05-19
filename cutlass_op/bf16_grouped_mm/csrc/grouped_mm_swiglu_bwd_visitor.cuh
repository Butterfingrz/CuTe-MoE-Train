// Copyright (c) Butterfingrz，13524387014@163.com

#pragma once

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>

namespace cutlass::epilogue::fusion {

// Arguments for the SwiGLU backward compute functor.
// Defined outside the template so SwiGLUBwdOp<float>::Arguments
// and SwiGLUBwdOp<Array<float,N>>::Arguments are the same type.
struct SwiGLUBwdArgs {
  float alpha = 1.702f;
  float limit = 7.0f;
};

// Element-wise SwiGLU backward on packed BF16 pairs.
//
// The GEMM computes: grad_swiglu = grad_output @ W_down^T  (shape [M, I])
// The C-load provides: packed aux_out, where each FP32 word packs two BF16
// values (gate, up) from the original [M, 2*I] tensor.
//
// This functor unpacks, computes d_gate and d_up in FP32, and repacks
// into an FP32 word for D-store. The D output, when reinterpreted as BF16,
// gives the interleaved grad_gate_up [M, 2*I].
//
// Usage with Sm90Compute:
//   using Compute = Sm90Compute<SwiGLUBwdOp, float, float, RoundStyle>;
//   using FusionOp = Sm90EVT<Compute, Sm90AccFetch, Sm90SrcFetch<float>>;
template <class T>
struct SwiGLUBwdOp {
  using Arguments = SwiGLUBwdArgs;

  CUTLASS_DEVICE static float
  compute_element(float grad_s, float packed_aux, float alpha, float limit) {
    // Unpack BF16 pair from FP32 word
    auto const* aux_bf16 = reinterpret_cast<__nv_bfloat16 const*>(&packed_aux);
    float gate = __bfloat162float(aux_bf16[0]);
    float up   = __bfloat162float(aux_bf16[1]);

    // Clamp (matching forward pass)
    float gate_c = fminf(gate, limit);
    float up_c   = fmaxf(fminf(up, limit), -limit);

    // sigmoid(alpha * gate_c) via tanh: sig = 0.5 * tanh(alpha*gate_c/2) + 0.5
    float half_ag = 0.5f * alpha * gate_c;
    float t = tanhf(half_ag);
    float sig = 0.5f * t + 0.5f;
    float glu = gate_c * sig;

    // SwiGLU backward: y = glu * (up_c + 1)
    // d_gate_c = grad_s * (up_c + 1) * sig * (1 + alpha * gate_c * (1 - sig))
    // d_up_c   = grad_s * glu
    float d_gate = grad_s * (up_c + 1.0f) * sig * (1.0f + alpha * gate_c * (1.0f - sig));
    float d_up   = grad_s * glu;

    // Clamp backward: zero grad where input was clamped
    d_gate *= (gate <= limit) ? 1.0f : 0.0f;
    d_up   *= (up >= -limit && up <= limit) ? 1.0f : 0.0f;

    // Pack BF16 pair back into FP32 word
    float packed_out;
    auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed_out);
    out_bf16[0] = __float2bfloat16_rn(d_gate);
    out_bf16[1] = __float2bfloat16_rn(d_up);
    return packed_out;
  }

  CUTLASS_DEVICE T operator()(T const& grad_s, T const& packed_aux,
                               Arguments const& args) const {
    T result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) {
      result[i] = compute_element(grad_s[i], packed_aux[i], args.alpha, args.limit);
    }
    return result;
  }
};

} // namespace cutlass::epilogue::fusion
