# CuTe-MoE-Train

## Overview

CuTe-MoE-Train is a collection of high-performance CUDA kernels and PyTorch
integrations targeting Mixture-of-Experts (MoE) training on NVIDIA Hopper
(SM90) GPUs.

## Modules

### `cutlass_op/bf16_grouped_mm`

A CUTLASS-based BF16 grouped GEMM kernel for SM90, integrated into PyTorch as
a custom op. On top of the standard grouped GEMM, this module further extends
the epilogue with:

- **Bias epilogue** — per-group bias broadcast fused into the GEMM epilogue,
  avoiding a separate elementwise pass.
- **SwiGLU epilogue** — fused SwiGLU activation (both forward and backward
  visitors) so the gate/up projections of an MoE expert can be produced and
  activated in a single grouped GEMM launch.

The kernel reuses CUTLASS's SM90 grouped GEMM infrastructure (TMA, warp-
specialized mainloop, persistent scheduler) and plugs the custom epilogues in
via CUTLASS EVT (Epilogue Visitor Tree) traits.

## Quick Start

The kernels target SM90 (Hopper) with bf16 inputs. See
`cutlass_op/bf16_grouped_mm/pixi.toml` and `cutlass_op/bf16_grouped_mm/setup.py`
for the build setup; once the extension is built, the example below runs
end-to-end on a single SM90 GPU.

The snippet below exercises the public API exported from
`cutlass_op/bf16_grouped_mm/__init__.py` and demonstrates the typical MoE FFN
flow — a fused up-projection (`grouped_mm_swiglu`) followed by a fused
down-projection (`grouped_mm_downproj`):

```python
import torch
from cutlass_op.bf16_grouped_mm import grouped_mm_swiglu, grouped_mm_downproj

device, dtype = "cuda", torch.bfloat16
E, H, I = 8, 2048, 1024                                  # 8 experts
tokens_per_expert = torch.tensor(                        # ragged-M distribution
    [128, 64, 0, 256, 192, 32, 128, 64], device=device, dtype=torch.int32
)
offs = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
total_M = int(tokens_per_expert.sum())

x      = torch.randn(total_M, H,  device=device, dtype=dtype, requires_grad=True)
W_up   = torch.randn(E, H, 2 * I, device=device, dtype=dtype, requires_grad=True)
b_up   = torch.randn(E, 2 * I,    device=device, dtype=dtype, requires_grad=True)
W_down = torch.randn(E, I, H,     device=device, dtype=dtype, requires_grad=True)
b_down = torch.randn(E, H,        device=device, dtype=dtype, requires_grad=True)

# === Forward: two fused kernels — aux_biased_acc must be threaded through ===
swiglu_out, aux_biased_acc = grouped_mm_swiglu(
    x, W_up, offs=offs, bias=b_up,
    swiglu_alpha=1.702, swiglu_limit=7.0,
)
out = grouped_mm_downproj(
    swiglu_out, W_down, offs=offs, bias=b_down,
    aux_biased_acc=aux_biased_acc,                       # <- fused-only contract
    swiglu_alpha=1.702, swiglu_limit=7.0,
)
assert out.shape == (total_M, H)

# === Backward: autograd flows back through the fused aux route ===
out.sum().backward()
assert W_up.grad is not None and W_down.grad is not None
```

> Note: `grouped_mm_swiglu` is fused-only. The `aux_biased_acc` returned by
> the up-projection must be passed into `grouped_mm_downproj`; that handoff is
> what lets SwiGLU's backward stay fused. Standalone backward through
> `swiglu_out` is intentionally unsupported.

## Performance

Measured on a single **NVIDIA H200** under **FSDP** with **no expert
parallelism (no-EP)** and **24K** input tokens per rank. The baseline is the
MoE block from [torchtitan](https://github.com/pytorch/torchtitan), which
relies on PyTorch's built-in `torch._grouped_mm`.

| Stage        | Baseline (`torch._grouped_mm`, torchtitan) | This module |
| ------------ | -----------------------------------------: | ----------: |
| MoE forward  |                                      21 ms |       14 ms |
| MoE backward |                                      68 ms |       61 ms |

The forward speedup comes primarily from the fused **GEMM + bias + SwiGLU**
epilogue (`grouped_mm_swiglu`), which collapses three passes into one. The
backward improvement comes from routing the down-projection's gradient
through `aux_biased_acc`, so SwiGLU's backward stays fused with the down-proj
dgrad (see `grouped_mm_downproj` in Quick Start).

## Roadmap

- **More low-precision × activation fusions.** Extend the EVT epilogue
  catalog beyond the current BF16 + bias + SwiGLU combination to additional
  low-precision input/output formats (e.g. FP8) paired with the activation
  variants common in MoE FFNs.
- **CUTLASS-based compute/communication overlap.** Build CUTLASS kernels
  that interleave collectives (all-to-all / reduce-scatter for EP, all-gather
  / reduce-scatter for FSDP) with the grouped GEMM mainloop, so MoE training
  can hide communication behind compute at the kernel level instead of at
  the Python/stream level.
- **Broader backend and architecture coverage.** Add alternative kernel
  implementations via **CuTe DSL** and **TileLang** alongside the existing
  CUTLASS C++ path, and extend architecture support to **SM80** (Ampere),
  **SM90** (Hopper, current), and **SM100** (Blackwell).

## Acknowledgements

This project is built on top of, and would not be possible without, the
following open-source projects:

- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — CUTLASS templates and
  CuTe abstractions that power the underlying SM90 grouped GEMM and EVT
  epilogues.
- [pytorch/pytorch](https://github.com/pytorch/pytorch) — PyTorch's custom op
  and extension infrastructure used to expose these kernels to Python.
