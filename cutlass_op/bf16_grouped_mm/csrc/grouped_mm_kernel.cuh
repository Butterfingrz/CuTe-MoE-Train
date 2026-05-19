// This file is adapted from https://github.com/pytorch/pytorch/blob/4a60bdc6b48caf621407de0ba28f2420024eab32/aten/src/ATen/native/cuda/GroupMM.cu

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/functional.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "cutlass_utils.cuh"
#include "grouped_mm_common.cuh"
#include "grouped_mm_swiglu_bwd_visitor.cuh"
#include "grouped_mm_swiglu_visitor.cuh"
#include "grouped_mm_row_broadcast_vec.cuh"
#include "grouped_mm_epilogue_traits.cuh"

template <bool PONGOr2SM, typename TB_M, typename TB_N, typename TB_K>
struct Schedule {
  // SM90 only - Hopper architecture scheduling policies
  using CooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;

  using KernelSchedule = cute::conditional_t<PONGOr2SM, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::conditional_t<PONGOr2SM, PongEpilogueSchedule, CooperativeEpilogueSchedule>;
};

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

inline int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

// Common type definitions shared by bias-only and SwiGLU grouped GEMM kernels
template <
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K,
    typename ClusterShape_>
struct GroupedGemmTypes {
  using DtypeA      = cutlass::bfloat16_t;
  using DtypeB      = cutlass::bfloat16_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using DtypeAccum  = float;
  using Strides     = at::cuda::detail::Strides;

  using LayoutA = cute::conditional_t<
      a_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  static constexpr int AlignmentA = 16 / sizeof(DtypeA);

  using LayoutB = cute::conditional_t<
      b_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  static constexpr int AlignmentB = 16 / sizeof(DtypeB);

  using LayoutOutput = cutlass::layout::RowMajor;
  static constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  using OperatorClass    = cutlass::arch::OpClassTensorOp;
  using TileShape        = cute::Shape<TB_M, TB_N, TB_K>;
  using ClusterShape     = ClusterShape_;
  using KernelSchedule   = typename Schedule<PONGOr2SM, TB_M, TB_N, TB_K>::KernelSchedule;
  using EpilogueSchedule = typename Schedule<PONGOr2SM, TB_M, TB_N, TB_K>::EpilogueSchedule;
  using ProblemShape     = cutlass::gemm::GroupProblemShape<
      cute::Shape<int32_t, int32_t, int32_t>>;
};

// =============================================================================
// Bias-only grouped GEMM kernel: D = GEMM(A, B) + bias
// =============================================================================

template <
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K,
    typename ClusterShape_,
    int RequestedStagesD = 0,          // 0 = builder default min(EpiTiles,2); >0 = override
    int RequestedMainloopStages = 0,   // 0 = auto via StageCountAutoCarveout; >0 = fixed
    typename RequestedEpiTile_ = cutlass::epilogue::collective::EpilogueTileAuto>
void bf16bf16_grouped_gemm_impl_sm90(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  using Types = GroupedGemmTypes<a_row_major, b_row_major, PONGOr2SM, TB_M, TB_N, TB_K, ClusterShape_>;
  using DtypeA           = typename Types::DtypeA;
  using DtypeB           = typename Types::DtypeB;
  using DtypeOutput      = typename Types::DtypeOutput;
  using DtypeAccum       = typename Types::DtypeAccum;
  using Strides          = typename Types::Strides;
  using LayoutA          = typename Types::LayoutA;
  static constexpr int AlignmentA = Types::AlignmentA;
  using LayoutB          = typename Types::LayoutB;
  static constexpr int AlignmentB = Types::AlignmentB;
  using LayoutOutput     = typename Types::LayoutOutput;
  static constexpr int AlignmentOutput = Types::AlignmentOutput;
  using OperatorClass    = typename Types::OperatorClass;
  using TileShape        = typename Types::TileShape;
  using ClusterShape     = typename Types::ClusterShape;
  using KernelSchedule   = typename Types::KernelSchedule;
  using EpilogueSchedule = typename Types::EpilogueSchedule;
  using ProblemShape     = typename Types::ProblemShape;

  // EVT leaf nodes
  using Accum     = cutlass::epilogue::fusion::Sm90AccFetch;
  using BiasBcast = cutlass::epilogue::fusion::Sm90RowBroadcastVec<
      0, TileShape, cutlass::bfloat16_t*, DtypeAccum,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;

  // EVT compute: D = acc + bias (PtrArray mode, consumer-warp cp.async bias load)
  using AccPlusBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus, DtypeOutput, DtypeAccum,
      cutlass::FloatRoundStyle::round_to_nearest>;

  // EVT tree
  using AccPlusBiasFusion = cutlass::epilogue::fusion::Sm90EVT<
      AccPlusBias, Accum, BiasBcast>;

  using CollectiveEpilogue_Base =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          TileShape,
          ClusterShape,
          RequestedEpiTile_,
          DtypeAccum,
          DtypeAccum,
          void,
          void,
          0,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          EpilogueSchedule,
          AccPlusBiasFusion>::CollectiveOp;

  // Resolve effective StagesD: user override or builder default.
  static constexpr int BuilderStagesD = CollectiveEpilogue_Base::DispatchPolicy::StagesD;
  static constexpr int StagesD = (RequestedStagesD > 0) ? RequestedStagesD : BuilderStagesD;

  // Override epilogue DispatchPolicy::StagesD
  using CollectiveEpilogue = typename OverrideStagesD<CollectiveEpilogue_Base, StagesD>::type;

  using MainloopStages = typename MainloopStageCountPolicy<
      RequestedMainloopStages,
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>::type;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          MainloopStages,
          KernelSchedule>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GemmKernel = enable_3x_kernel_for_sm9x<GemmKernelBase>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  // Epilogue overhead diagnostics (bias-only kernel)
  {
    static bool printed = false;
    if (!printed) {
      printed = true;
      constexpr size_t epi_smem = sizeof(typename CollectiveEpilogue::SharedStorage);
      constexpr int mainloop_stages = CollectiveMainloop::DispatchPolicy::Stages;
      constexpr int stages_d = CollectiveEpilogue::DispatchPolicy::StagesD;
      fprintf(stderr,
          "[grouped_mm bias] tile=<%d,%d,%d> pong=%d | "
          "epilogue_smem=%zu B | mainloop_stages=%d | stages_d=%d\n",
          int(TB_M::value), int(TB_N::value), int(TB_K::value),
          int(PONGOr2SM), epi_smem, mainloop_stages, stages_d);
    }
  }

  int32_t M, N, K, group_count;

  M = mat_a.size(-2);
  K = mat_a.size(-1);
  N = mat_b.size(-1);

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    group_count = offs->size(0);  // metadata-only, no D2H
    K = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = -1;
  } else {
    // regular bmm
    group_count = mat_a.size(0);
  }

  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
  const int64_t problem_shape_size =
      group_count * ((int64_t)sizeof(typename ProblemShape::UnderlyingProblemShape));

  const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));
  // Align to 128 bits for CUDA <12.4 TMA bug workaround.
  // Dummy TMAs are created based on these pointer-to-pointers;
  // the actual values are never used, they are replaced
  // by real addresses, but for dummy tma creation to succeed
  // due to bug in cuda < 12.4 the pointers have to be aligned to 128 bits
  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);
  int64_t input_args_size = aligned_group_count * 3 * sizeof(void*) +
      problem_shape_size + stride_size;
  if (bias.has_value()) {
    input_args_size += aligned_group_count * sizeof(void*);
  }

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);

  cutlass::bfloat16_t** bias_ptrs = nullptr;
  if (bias.has_value()) {
    bias_ptrs = reinterpret_cast<cutlass::bfloat16_t**>(output_ptrs + aligned_group_count);
  }

  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A = reinterpret_cast<StrideA*>(
      bias.has_value() ?
      reinterpret_cast<void*>(bias_ptrs + aligned_group_count) :
      reinterpret_cast<void*>(output_ptrs + aligned_group_count));
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output =
      reinterpret_cast<StrideOutput*>(stride_B + group_count);
  typename ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
          stride_output + group_count);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto make_strides = [](at::IntArrayRef strides) -> Strides {
    Strides out;
    std::copy(strides.begin(), strides.end(), out.begin());
    return out;
  };

  Strides tensor_StrideA = make_strides(mat_a.strides());
  Strides tensor_StrideB = make_strides(mat_b.strides());
  Strides tensor_StrideOutput = make_strides(out.strides());
  Strides tensor_ShapeA = make_strides(mat_a.sizes());
  Strides tensor_ShapeB = make_strides(mat_b.sizes());

  Strides tensor_ShapeBias{};
  Strides tensor_StrideBias{};
  if (bias.has_value()) {
    tensor_ShapeBias = make_strides(bias->sizes());
    tensor_StrideBias = make_strides(bias->strides());
  }

  at::cuda::detail::prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(out.data_ptr()),
      static_cast<float*>(nullptr),
      static_cast<float*>(nullptr),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      static_cast<float**>(nullptr),
      static_cast<float**>(nullptr),
      bias.has_value() ? reinterpret_cast<cutlass::bfloat16_t*>(bias->data_ptr()) : nullptr,
      bias_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      M,
      N,
      K,
      tensor_StrideA,
      tensor_StrideB,
      tensor_StrideOutput,
      tensor_ShapeBias,
      tensor_StrideBias,
      tensor_ShapeA,
      tensor_ShapeB,
      0,
      0,
      a_row_major,
      b_row_major);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{},
       nullptr,
       nullptr,
       output_ptrs,
       stride_output}};

  int sm_count =
      at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  // EVT arguments: {AccFetch_args, RowBroadcastVec_args, Compute_args}
  arguments.epilogue.thread = {
      {},  // Accum (Sm90AccFetch) args: empty
      {    // BiasBcast (Sm90RowBroadcastVec) args: PtrArray mode
          bias.has_value() ?
              (const cutlass::bfloat16_t* const*)bias_ptrs : nullptr,
          cutlass::bfloat16_t(0),  // null_default
          {},                      // dRow stride
      },
      {}   // AccPlusBias (Sm90Compute<plus>) args: empty
  };

  arguments.hw_info.sm_count = sm_count;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass cannot run, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool a_row_major, bool b_row_major>
void dispatch_bf16_grouped_kernel_on_tile_size(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  int32_t M, N, K, group_count;

  M = mat_a.size(-2);
  K = mat_a.size(-1);
  N = mat_b.size(-1);

  // below we assume that gemms are approx same size
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // if both inputs are ragged, K is dynamic, M and N come from inputs
    group_count = offs->size(0);
    K = K / group_count;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = M / group_count;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = N / group_count;
  }

  // SM90 only: tile size selection based on problem size
  if (M <= 64) {
    bf16bf16_grouped_gemm_impl_sm90<
      a_row_major,
      b_row_major,
      /*PONGOr2SM*/ true,
      cute::_64,
      cute::_128,
      cute::_128,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      /*RequestedStagesD=*/0,
      /*RequestedMainloopStages=*/0>(mat_a, mat_b, offs, bias, out);
  } else if (M <= 128) {
    bf16bf16_grouped_gemm_impl_sm90<
      a_row_major,
      b_row_major,
      /*PONGOr2SM*/ true,
      cute::_128,
      cute::_128,
      cute::_64,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      /*RequestedStagesD=*/0,
      /*RequestedMainloopStages=*/0>(mat_a, mat_b, offs, bias, out);
  } else {
    bf16bf16_grouped_gemm_impl_sm90<
      a_row_major,
      b_row_major,
      /*PONGOr2SM*/ false,
      cute::_128,
      cute::_256,
      cute::_64,
      cute::Shape<cute::_2, cute::_1, cute::_1>,
      /*RequestedStagesD=*/0,
      /*RequestedMainloopStages=*/0>(mat_a, mat_b, offs, bias, out);
  }
}

void dispatch_bf16_grouped_kernel_on_ab_transpose(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  // we already checked that one of the strides is 1
  bool a_row_major = mat_a.stride(-1) == 1;
  bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<true, true>(
        mat_a, mat_b, offs, bias, out);
  } else if (a_row_major && !b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<true, false>(
        mat_a, mat_b, offs, bias, out);
  } else if (!a_row_major && b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<false, true>(
        mat_a, mat_b, offs, bias, out);
  } else {
    dispatch_bf16_grouped_kernel_on_tile_size<false, false>(
        mat_a, mat_b, offs, bias, out);
  }
}


void bf16bf16_grouped_mm(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  // SM90 (Hopper) only - requires H100/H200 GPU
  dispatch_bf16_grouped_kernel_on_ab_transpose(mat_a, mat_b, offs, bias, out);
}

// =============================================================================
// Fused down-proj dgrad + SwiGLU backward kernel
// =============================================================================
// Computes grad_gate_up in one kernel:
//   grad_swiglu = grad_output @ down_w_t
//   grad_gate_up = SwiGLU_bwd(grad_swiglu, aux_packed)
//
// Tensor contracts (ragged-M path):
//   grad_output      [total_M, H] bf16
//   down_w_t         [E, H, I] bf16
//   aux_packed       [total_M, I] fp32 (view of aux_out [total_M, 2*I] bf16)
//   grad_gate_up_out [total_M, I] fp32 (view of grad_gate_up [total_M, 2*I] bf16)

template <
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K,
    typename ClusterShape_,
    int RequestedStagesD = 0,          // 0 = builder default min(EpiTiles,2); >0 = override
    int RequestedMainloopStages = 0,   // 0 = auto via StageCountAutoCarveout; >0 = fixed
    typename RequestedEpiTile_ = cutlass::epilogue::collective::EpilogueTileAuto>
void bf16bf16_grouped_gemm_swiglu_bwd_impl_sm90(
    at::Tensor grad_output,             // [total_M, H] bf16
    at::Tensor down_w_t,                // [E, H, I] bf16
    at::Tensor offs_tensor,             // [E] int32 offsets
    at::Tensor aux_packed,              // [total_M, I] fp32 packed aux
    at::Tensor& grad_gate_up_packed,    // [total_M, I] fp32 packed dgate/dup
    float swiglu_alpha,
    float swiglu_limit) {
  using Types = GroupedGemmTypes<a_row_major, b_row_major, PONGOr2SM, TB_M, TB_N, TB_K, ClusterShape_>;
  using DtypeA      = typename Types::DtypeA;
  using DtypeB      = typename Types::DtypeB;
  using DtypeAccum  = typename Types::DtypeAccum;
  using Strides     = typename Types::Strides;
  using LayoutA     = typename Types::LayoutA;
  using LayoutB     = typename Types::LayoutB;
  using LayoutOutput = typename Types::LayoutOutput;
  using OperatorClass    = typename Types::OperatorClass;
  using TileShape        = typename Types::TileShape;
  using ClusterShape     = typename Types::ClusterShape;
  using KernelSchedule   = typename Types::KernelSchedule;
  using EpilogueSchedule = typename Types::EpilogueSchedule;
  using ProblemShape     = typename Types::ProblemShape;

  using DtypePacked = float;
  static constexpr int AlignmentA = Types::AlignmentA;
  static constexpr int AlignmentB = Types::AlignmentB;
  static constexpr int AlignmentPacked = 16 / sizeof(DtypePacked);
  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using SrcLoad = cutlass::epilogue::fusion::Sm90SrcFetch<DtypePacked>;
  using SwiGLUBwdCompute = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::fusion::SwiGLUBwdOp,
      DtypePacked,
      DtypePacked,
      RoundStyle>;
  using SwiGLUBwdFusion = cutlass::epilogue::fusion::Sm90EVT<
      SwiGLUBwdCompute, Accum, SrcLoad>;

  using CollectiveEpilogue_Base =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          TileShape,
          ClusterShape,
          RequestedEpiTile_,
          DtypeAccum,
          DtypeAccum,
          DtypePacked,
          LayoutOutput*,
          AlignmentPacked,
          DtypePacked,
          LayoutOutput*,
          AlignmentPacked,
          EpilogueSchedule,
          SwiGLUBwdFusion>::CollectiveOp;

  static constexpr int BuilderStagesD = CollectiveEpilogue_Base::DispatchPolicy::StagesD;
  static constexpr int StagesD = (RequestedStagesD > 0) ? RequestedStagesD : BuilderStagesD;
  using CollectiveEpilogue = typename OverrideStagesD<CollectiveEpilogue_Base, StagesD>::type;

  using MainloopStages = typename MainloopStageCountPolicy<
      RequestedMainloopStages,
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>::type;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          MainloopStages,
          KernelSchedule>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GemmKernel = enable_3x_kernel_for_sm9x<GemmKernelBase>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int32_t group_count = down_w_t.size(0);
  int32_t M = -1;                        // ragged-M
  int32_t K = grad_output.size(-1);      // H
  int32_t N = down_w_t.size(-1);         // I

  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");

  const int64_t problem_shape_size =
      group_count * ((int64_t)sizeof(typename ProblemShape::UnderlyingProblemShape));

  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);

  // Pointer arrays: A, B, C(aux), D(grad_gate_up)
  int64_t input_args_size = aligned_group_count * 4 * sizeof(void*) + problem_shape_size;
  input_args_size += group_count * sizeof(StrideA);
  input_args_size += group_count * sizeof(StrideB);
  input_args_size += group_count * sizeof(StrideC);
  input_args_size += group_count * sizeof(StrideD);

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypePacked** inputC_ptrs =
      reinterpret_cast<DtypePacked**>(inputB_ptrs + aligned_group_count);
  DtypePacked** output_ptrs =
      reinterpret_cast<DtypePacked**>(inputC_ptrs + aligned_group_count);

  StrideA* stride_A = reinterpret_cast<StrideA*>(output_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideC* stride_C = reinterpret_cast<StrideC*>(stride_B + group_count);
  StrideD* stride_D = reinterpret_cast<StrideD*>(stride_C + group_count);
  typename ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(stride_D + group_count);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto make_strides = [](at::IntArrayRef strides) -> Strides {
    Strides out;
    std::copy(strides.begin(), strides.end(), out.begin());
    return out;
  };

  Strides tensor_StrideA = make_strides(grad_output.strides());
  Strides tensor_StrideB = make_strides(down_w_t.strides());
  Strides tensor_StrideOutput = make_strides(grad_gate_up_packed.strides());
  Strides tensor_ShapeA = make_strides(grad_output.sizes());
  Strides tensor_ShapeB = make_strides(down_w_t.sizes());
  Strides tensor_ShapeBias{};
  Strides tensor_StrideBias{};

  // Use existing grouped pointer preparation path for A/B/D and per-group problem shapes.
  at::cuda::detail::prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(grad_output.data_ptr()),
      reinterpret_cast<DtypeB*>(down_w_t.data_ptr()),
      reinterpret_cast<DtypePacked*>(grad_gate_up_packed.data_ptr()),
      static_cast<float*>(nullptr),
      static_cast<float*>(nullptr),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      static_cast<float**>(nullptr),
      static_cast<float**>(nullptr),
      static_cast<cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t**>(nullptr),
      problem_sizes,
      stride_A,
      stride_B,
      stride_D,
      offs_tensor.const_data_ptr<int32_t>(),
      M,
      N,
      K,
      tensor_StrideA,
      tensor_StrideB,
      tensor_StrideOutput,
      tensor_ShapeBias,
      tensor_StrideBias,
      tensor_ShapeA,
      tensor_ShapeB,
      0,
      0,
      a_row_major,
      b_row_major);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Prepare C(src) pointer array from packed aux tensor.
  int32_t stride_aux = static_cast<int32_t>(aux_packed.stride(0));
  at::cuda::detail::prepare_swiglu_ptrs<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypePacked*>(aux_packed.data_ptr()),
      inputC_ptrs,
      offs_tensor.const_data_ptr<int32_t>(),
      stride_aux);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // C and D share the same [M, N, L] stride topology in this kernel.
  cudaError_t stride_copy_status = cudaMemcpyAsync(
      stride_C,
      stride_D,
      group_count * sizeof(StrideC),
      cudaMemcpyDeviceToDevice,
      stream);
  TORCH_CHECK(
      stride_copy_status == cudaSuccess,
      "cudaMemcpyAsync stride copy failed: ",
      cudaGetErrorString(stride_copy_status));

  int sm_count =
      at::cuda::getDeviceProperties(grad_output.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{},
       (const DtypePacked**)inputC_ptrs,
       stride_C,
       output_ptrs,
       stride_D}};

  // EVT args order is leaf-to-root: {AccFetch, SrcFetch, Compute}.
  arguments.epilogue.thread = {
      {},  // Accum (Sm90AccFetch)
      {},  // C-load source (Sm90SrcFetch<float>)
      {swiglu_alpha, swiglu_limit}  // SwiGLUBwdOp args
  };
  arguments.hw_info.sm_count = sm_count;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement swiglu_bwd fused kernel");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize swiglu_bwd fused kernel");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass swiglu_bwd fused kernel failed, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool a_row_major, bool b_row_major>
void dispatch_bf16_grouped_swiglu_bwd_on_tile_size(
    at::Tensor grad_output,
    at::Tensor down_w_t,
    at::Tensor offs,
    at::Tensor aux_packed,
    at::Tensor& grad_gate_up_packed,
    float swiglu_alpha,
    float swiglu_limit) {
  int32_t group_count = down_w_t.size(0);
  int32_t avg_m = grad_output.size(0) / group_count;

  if (avg_m <= 64) {
    bf16bf16_grouped_gemm_swiglu_bwd_impl_sm90<
        a_row_major,
        b_row_major,
        /*PONGOr2SM=*/true,
        cute::_64,
        cute::_128,
        cute::_128,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        /*RequestedStagesD=*/0,
        /*RequestedMainloopStages=*/0>(
        grad_output,
        down_w_t,
        offs,
        aux_packed,
        grad_gate_up_packed,
        swiglu_alpha,
        swiglu_limit);
  } else if (avg_m <= 128) {
    bf16bf16_grouped_gemm_swiglu_bwd_impl_sm90<
        a_row_major,
        b_row_major,
        /*PONGOr2SM=*/true,
        cute::_128,
        cute::_128,
        cute::_64,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        /*RequestedStagesD=*/0,
        /*RequestedMainloopStages=*/0>(
        grad_output,
        down_w_t,
        offs,
        aux_packed,
        grad_gate_up_packed,
        swiglu_alpha,
        swiglu_limit);
  } else {
    bf16bf16_grouped_gemm_swiglu_bwd_impl_sm90<
        a_row_major,
        b_row_major,
        /*PONGOr2SM=*/false,
        cute::_128,
        cute::_256,
        cute::_64,
        cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*RequestedStagesD=*/0,
        /*RequestedMainloopStages=*/0>(
        grad_output,
        down_w_t,
        offs,
        aux_packed,
        grad_gate_up_packed,
        swiglu_alpha,
        swiglu_limit);
  }
}

void dispatch_bf16_grouped_swiglu_bwd_on_ab_transpose(
    at::Tensor grad_output,
    at::Tensor down_w_t,
    at::Tensor offs,
    at::Tensor aux_packed,
    at::Tensor& grad_gate_up_packed,
    float swiglu_alpha,
    float swiglu_limit) {
  bool a_row_major = grad_output.stride(-1) == 1;
  bool b_row_major = down_w_t.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_bf16_grouped_swiglu_bwd_on_tile_size<true, true>(
        grad_output, down_w_t, offs, aux_packed, grad_gate_up_packed, swiglu_alpha, swiglu_limit);
  } else if (a_row_major && !b_row_major) {
    dispatch_bf16_grouped_swiglu_bwd_on_tile_size<true, false>(
        grad_output, down_w_t, offs, aux_packed, grad_gate_up_packed, swiglu_alpha, swiglu_limit);
  } else if (!a_row_major && b_row_major) {
    dispatch_bf16_grouped_swiglu_bwd_on_tile_size<false, true>(
        grad_output, down_w_t, offs, aux_packed, grad_gate_up_packed, swiglu_alpha, swiglu_limit);
  } else {
    dispatch_bf16_grouped_swiglu_bwd_on_tile_size<false, false>(
        grad_output, down_w_t, offs, aux_packed, grad_gate_up_packed, swiglu_alpha, swiglu_limit);
  }
}

void bf16bf16_grouped_mm_swiglu_bwd(
    at::Tensor grad_output,          // [M, H] bf16
    at::Tensor down_w_t,             // [E, H, I] bf16
    at::Tensor offs,                 // [E] int32
    at::Tensor aux_packed,           // [M, I] fp32
    at::Tensor& grad_gate_up_packed, // [M, I] fp32
    float swiglu_alpha,
    float swiglu_limit) {
  dispatch_bf16_grouped_swiglu_bwd_on_ab_transpose(
      grad_output, down_w_t, offs, aux_packed, grad_gate_up_packed, swiglu_alpha, swiglu_limit);
}

// =============================================================================
// Fused GEMM + Bias + Gated-SwiGLU kernel
// =============================================================================
// Supports two modes:
//   StoreAuxD=true  -> produce aux_out [total_M, 2*I] via D store and swiglu_out [total_M, I]
//   StoreAuxD=false -> skip aux D store, only produce swiglu_out

template <
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K,
    typename ClusterShape_,
    int RequestedStagesD = 0,          // 0 = builder default min(EpiTiles,2); >0 = override
    int RequestedMainloopStages = 0,   // 0 = auto via StageCountAutoCarveout; >0 = fixed
    typename RequestedEpiTile_ = cutlass::epilogue::collective::EpilogueTileAuto,
    bool StoreAuxD = true>
void bf16bf16_grouped_gemm_swiglu_impl_sm90(
    at::Tensor mat_a,              // [total_M, K]
    at::Tensor mat_b,              // [E, K, 2*I]
    at::Tensor offs_tensor,        // [E] cumsum offsets (required)
    at::Tensor bias_tensor,        // [E, 2*I] (required)
    const std::optional<at::Tensor>& aux_out_opt, // [total_M, 2*I] when StoreAuxD=true
    at::Tensor& swiglu_out,        // [total_M, I] SwiGLU output
    float swiglu_alpha,
    float swiglu_limit) {

  using Types = GroupedGemmTypes<a_row_major, b_row_major, PONGOr2SM, TB_M, TB_N, TB_K, ClusterShape_>;
  using DtypeA           = typename Types::DtypeA;
  using DtypeB           = typename Types::DtypeB;
  using DtypeOutput      = typename Types::DtypeOutput;
  using DtypeAccum       = typename Types::DtypeAccum;
  using Strides          = typename Types::Strides;
  using LayoutA          = typename Types::LayoutA;
  static constexpr int AlignmentA = Types::AlignmentA;
  using LayoutB          = typename Types::LayoutB;
  static constexpr int AlignmentB = Types::AlignmentB;
  using LayoutOutput     = typename Types::LayoutOutput;
  static constexpr int AlignmentOutput = Types::AlignmentOutput;
  using OperatorClass    = typename Types::OperatorClass;
  using TileShape        = typename Types::TileShape;
  using ClusterShape     = typename Types::ClusterShape;
  using KernelSchedule   = typename Types::KernelSchedule;
  using EpilogueSchedule = typename Types::EpilogueSchedule;
  using ProblemShape     = typename Types::ProblemShape;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr int AlignBias = 16 / sizeof(cutlass::bfloat16_t);

  // --- Phase 1: Build reference epilogue to extract compile-time parameters ---
  // The SwiGLU visitor needs EpilogueTile, StagesD, SmemLayoutAtomD, CopyOpR2S
  // from the builder, but these are determined internally by CollectiveBuilder.
  // Build a reference epilogue with simple bias fusion to extract them.

  // Reference EVT leaf nodes
  using Accum_Ref     = cutlass::epilogue::fusion::Sm90AccFetch;
  using BiasBcast_Ref = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, cutlass::bfloat16_t*, DtypeAccum,
      cute::Stride<cute::_0, cute::_1, int64_t>, AlignBias>;

  // Reference EVT compute
  using AccPlusBias_Ref = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus, DtypeOutput, DtypeAccum, RoundStyle>;

  // Reference EVT tree
  using AccPlusBiasFusion_Ref = cutlass::epilogue::fusion::Sm90EVT<
      AccPlusBias_Ref, Accum_Ref, BiasBcast_Ref>;

  using RefEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, OperatorClass, TileShape, ClusterShape,
          RequestedEpiTile_,
          DtypeAccum, DtypeAccum, void, void, 0,
          DtypeOutput, LayoutOutput*, AlignmentOutput,
          EpilogueSchedule, AccPlusBiasFusion_Ref>::CollectiveOp;

  // Extract epilogue parameters from the reference build
  using EpiTile         = typename RefEpilogue::EpilogueTile;
  using SmemLayoutAtomD = typename RefEpilogue::SmemLayoutAtomD;
  using CopyOpR2S_D     = typename RefEpilogue::CopyOpR2S;

  // Resolve effective StagesD: user override or builder default.
  // Must be computed before constructing SwiGLUFusion so visitor smem matches dispatch policy.
  static constexpr int BuilderStagesD = RefEpilogue::DispatchPolicy::StagesD;
  static constexpr int EpiTilesCount = cute::size(cute::shape_div(
      cute::take<0,2>(TileShape{}), EpiTile{}));
  static constexpr int StagesD = (RequestedStagesD > 0) ? RequestedStagesD : BuilderStagesD;
  static_assert(StagesD >= 1 && StagesD <= EpiTilesCount,
      "StagesD must be in [1, EpiTilesCount]");

  // --- Phase 2: Build the real SwiGLU fusion with extracted params ---
  static constexpr int EpiThreadCount = PONGOr2SM ? 128 : 256;

  // Child EVT leaf nodes: acc + bias (consumer-warp cp.async, no producer warp overhead)
  using Accum_SW     = cutlass::epilogue::fusion::Sm90AccFetch;
  using BiasBcast_SW = cutlass::epilogue::fusion::Sm90RowBroadcastVec<
      0, TileShape, cutlass::bfloat16_t*, DtypeAccum,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;

  // Child EVT compute
  using AccPlusBias_SW = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus, DtypeOutput, DtypeAccum, RoundStyle>;

  // Child EVT tree: acc + bias
  using AccPlusBiasEVT = cutlass::epilogue::fusion::Sm90EVT<
      AccPlusBias_SW, Accum_SW, BiasBcast_SW>;

  // Root EVT node: SwiGLU store with TMA pipeline
  // Accumulates SwiGLU in registers, R2S to smem, TMA to gmem.
  // Returns frg_input unchanged for D store (writes full [M, 2I] for backward).
  using SwiGLUStore = cutlass::epilogue::fusion::Sm90GatedSwiGLUStoreTma<
      StagesD, EpiTile, DtypeOutput, DtypeAccum, RoundStyle,
      SmemLayoutAtomD, CopyOpR2S_D, AlignmentOutput, true, EpiThreadCount>;

  // Complete EVT tree: SwiGLUStore( AccPlusBias( Accum, BiasBcast ) )
  using SwiGLUFusion = cutlass::epilogue::fusion::Sm90EVT<
      SwiGLUStore, AccPlusBiasEVT>;

  using CollectiveEpilogue_Base =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          TileShape,
          ClusterShape,
          RequestedEpiTile_,
          DtypeAccum,
          DtypeAccum,
          void,
          void,
          0,
          cute::conditional_t<StoreAuxD, DtypeOutput, void>,
          LayoutOutput*,
          StoreAuxD ? AlignmentOutput : 1,
          EpilogueSchedule,
          SwiGLUFusion>::CollectiveOp;

  // Override epilogue DispatchPolicy::StagesD to match the effective StagesD used
  // in the SwiGLU visitor. The builder hardcodes StagesD = min(EpiTiles, 2).
  using CollectiveEpilogue_StagesOverride = typename OverrideStagesD<CollectiveEpilogue_Base, StagesD>::type;
  using CollectiveEpilogue = CollectiveEpilogue_StagesOverride;

  // Compile-time diagnostics for SwiGLU visitor layout
  using SwiGLUVisitor = SwiGLUStore;
  using ActualEpiTile = typename CollectiveEpilogue::EpilogueTile;
  static_assert(cute::size<0>(EpiTile{}) == cute::size<0>(ActualEpiTile{}),
      "EpiTile M mismatch between RefEpilogue and SwiGLU CollectiveEpilogue");
  static_assert(cute::size<1>(EpiTile{}) == cute::size<1>(ActualEpiTile{}),
      "EpiTile N mismatch between RefEpilogue and SwiGLU CollectiveEpilogue");
  static_assert(cute::cosize(typename SwiGLUVisitor::SmemLayoutTma{})
             >= cute::size(typename SwiGLUVisitor::SmemLayoutTma{}),
      "SmemLayoutTma cosize must be >= size");

  using MainloopStages = typename MainloopStageCountPolicy<
      RequestedMainloopStages,
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>::type;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          MainloopStages,
          KernelSchedule>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GemmKernel = enable_3x_kernel_for_sm9x<GemmKernelBase>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  // Epilogue overhead diagnostics (SwiGLU fused kernel)
  {
    static bool printed = false;
    if (!printed) {
      printed = true;
      constexpr size_t epi_smem = sizeof(typename CollectiveEpilogue::SharedStorage);
      constexpr int mainloop_stages = CollectiveMainloop::DispatchPolicy::Stages;
      constexpr int stages_d = CollectiveEpilogue::DispatchPolicy::StagesD;
      constexpr int epi_m = cute::size<0>(EpiTile{});
      constexpr int epi_n = cute::size<1>(EpiTile{});
      constexpr int swiglu_smem = sizeof(typename SwiGLUVisitor::SharedStorage);
      constexpr int results_per_thread = epi_m * (epi_n / 2) / SwiGLUVisitor::ThreadCount;
      constexpr bool delay_tma = CollectiveEpilogue::DispatchPolicy::DelayTmaStore;
      fprintf(stderr,
          "[grouped_mm swiglu] tile=<%d,%d,%d> pong=%d | "
          "epilogue_smem=%zu B (swiglu_visitor=%d B) | "
          "mainloop_stages=%d | stages_d=%d | epi_tile=<%d,%d> | "
          "results_per_thread=%d | delay_tma_store=%d\n",
          int(TB_M::value), int(TB_N::value), int(TB_K::value),
          int(PONGOr2SM), epi_smem, swiglu_smem,
          mainloop_stages, stages_d, epi_m, epi_n, results_per_thread,
          int(delay_tma));
    }
  }

  // This kernel only supports ragged-M path (mat_a is 2D, mat_b is 3D)
  int32_t group_count = mat_b.size(0);
  int32_t M = -1;  // ragged
  int32_t K = mat_a.size(-1);
  int32_t N = mat_b.size(-1);  // = 2 * intermediate_size

  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");

  const int64_t problem_shape_size =
      group_count * ((int64_t)sizeof(typename ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));

  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);

  // Pointer arrays: A, B, D(=aux_out) + bias + swiglu
  int64_t input_args_size = aligned_group_count * 3 * sizeof(void*) +
      problem_shape_size + stride_size;
  input_args_size += aligned_group_count * sizeof(void*);  // bias_ptrs
  input_args_size += aligned_group_count * sizeof(void*);  // swiglu_ptrs

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);
  cutlass::bfloat16_t** bias_ptrs =
      reinterpret_cast<cutlass::bfloat16_t**>(output_ptrs + aligned_group_count);
  DtypeOutput** swiglu_ptrs =
      reinterpret_cast<DtypeOutput**>(bias_ptrs + aligned_group_count);

  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A = reinterpret_cast<StrideA*>(
      swiglu_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output =
      reinterpret_cast<StrideOutput*>(stride_B + group_count);
  typename ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
          stride_output + group_count);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto make_strides = [](at::IntArrayRef strides) -> Strides {
    Strides out;
    std::copy(strides.begin(), strides.end(), out.begin());
    return out;
  };

  at::Tensor aux_for_meta;
  if constexpr (StoreAuxD) {
    TORCH_CHECK(aux_out_opt.has_value(), "StoreAuxD=true requires aux_out tensor");
    aux_for_meta = *aux_out_opt;
  } else {
    aux_for_meta = at::empty({1, 1}, swiglu_out.options());
  }

  Strides tensor_StrideA = make_strides(mat_a.strides());
  Strides tensor_StrideB = make_strides(mat_b.strides());
  Strides tensor_StrideOutput = make_strides(aux_for_meta.strides());
  Strides tensor_ShapeA = make_strides(mat_a.sizes());
  Strides tensor_ShapeB = make_strides(mat_b.sizes());
  Strides tensor_ShapeBias = make_strides(bias_tensor.sizes());
  Strides tensor_StrideBias = make_strides(bias_tensor.strides());

  at::cuda::detail::prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(aux_for_meta.data_ptr()),
      static_cast<float*>(nullptr),
      static_cast<float*>(nullptr),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      static_cast<float**>(nullptr),
      static_cast<float**>(nullptr),
      reinterpret_cast<cutlass::bfloat16_t*>(bias_tensor.data_ptr()),
      bias_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs_tensor.const_data_ptr<int32_t>(),
      M,
      N,
      K,
      tensor_StrideA,
      tensor_StrideB,
      tensor_StrideOutput,
      tensor_ShapeBias,
      tensor_StrideBias,
      tensor_ShapeA,
      tensor_ShapeB,
      0,
      0,
      a_row_major,
      b_row_major);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Prepare swiglu output pointer array
  int32_t stride_swiglu = static_cast<int32_t>(swiglu_out.stride(0));
  at::cuda::detail::prepare_swiglu_ptrs<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeOutput*>(swiglu_out.data_ptr()),
      swiglu_ptrs,
      offs_tensor.const_data_ptr<int32_t>(),
      stride_swiglu);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int sm_count =
      at::cuda::getDeviceProperties(swiglu_out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  typename Gemm::Arguments arguments{};
  if constexpr (StoreAuxD) {
    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {group_count, problem_sizes, nullptr},
        {(const DtypeA**)inputA_ptrs,
         stride_A,
         (const DtypeB**)inputB_ptrs,
         stride_B},
        {{},
         nullptr,
         nullptr,
         output_ptrs,     // D store -> aux_out [total_M, 2*I]
         stride_output}};
  } else {
    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {group_count, problem_sizes, nullptr},
        {(const DtypeA**)inputA_ptrs,
         stride_A,
         (const DtypeB**)inputB_ptrs,
         stride_B},
        {{},
         nullptr,
         nullptr,
         nullptr,
         static_cast<decltype(stride_output)>(nullptr)}};
  }

  // EVT arguments: {child_EVT_args, root_node_args}
  // child = AccPlusBiasEVT: {Accum_args, BiasBcast_args, AccPlusBias_args}
  // root  = SwiGLUStore args
  arguments.epilogue.thread = {
      {   // Child: AccPlusBiasEVT
          {},  // Accum_SW (Sm90AccFetch) args: empty
          {    // BiasBcast_SW (Sm90RowBroadcastVec) args: PtrArray mode
              (const cutlass::bfloat16_t* const*)bias_ptrs,
              cutlass::bfloat16_t(0),
              {},  // dRow stride
          },
          {}   // AccPlusBias_SW (Sm90Compute<plus>) args: empty
      },
      {   // Root: SwiGLUStore args
          swiglu_ptrs,
          static_cast<int32_t>(N / 2),  // N_out
          stride_swiglu,
          swiglu_alpha,
          swiglu_limit
      }
  };

  arguments.hw_info.sm_count = sm_count;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement swiglu fused kernel");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize swiglu fused kernel");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass swiglu fused kernel failed, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Dispatch on tile size for swiglu fused kernel
template <bool a_row_major, bool b_row_major>
void dispatch_bf16_grouped_swiglu_on_tile_size(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor bias,
    const std::optional<at::Tensor>& aux_out_opt,
    at::Tensor& swiglu_out,
    float swiglu_alpha,
    float swiglu_limit,
    bool store_aux) {
  int32_t M, N;

  // Ragged-M path: mat_a is 2D, mat_b is 3D
  int32_t group_count = mat_b.size(0);
  M = mat_a.size(0) / group_count;  // approximate average M per group
  N = mat_b.size(-1);

  if (M <= 64) {
    if (store_aux) {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
        a_row_major, b_row_major,
        /*PONGOr2SM*/ true,
        cute::_64, cute::_128, cute::_128,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        /*RequestedStagesD=*/0,
        /*RequestedMainloopStages=*/0,
        cute::Shape<cute::_64, cute::_32>,
        /*StoreAuxD=*/true>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    } else {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
        a_row_major, b_row_major,
        /*PONGOr2SM*/ true,
        cute::_64, cute::_128, cute::_128,
        cute::Shape<cute::_1, cute::_1, cute::_1>,
        /*RequestedStagesD=*/0,
        /*RequestedMainloopStages=*/0,
        cute::Shape<cute::_64, cute::_32>,
        /*StoreAuxD=*/false>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    }
  } else if (M <= 128) {
    if (store_aux) {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
          a_row_major, b_row_major, /*PONGOr2SM=*/true,
          cute::_128, cute::_128, cute::_64,
          cute::Shape<cute::_1, cute::_1, cute::_1>,
          /*RequestedStagesD=*/0,
          /*RequestedMainloopStages=*/0,
          cute::Shape<cute::_64, cute::_32>,
          /*StoreAuxD=*/true>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    } else {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
          a_row_major, b_row_major, /*PONGOr2SM=*/true,
          cute::_128, cute::_128, cute::_64,
          cute::Shape<cute::_1, cute::_1, cute::_1>,
          /*RequestedStagesD=*/0,
          /*RequestedMainloopStages=*/0,
          cute::Shape<cute::_64, cute::_32>,
          /*StoreAuxD=*/false>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    }
  } else {
    if (store_aux) {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
        a_row_major, b_row_major,
        /*PONGOr2SM*/ false,
        cute::_128, cute::_256, cute::_64,
        cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*RequestedStagesD=*/2,
        /*RequestedMainloopStages=*/4,
        cute::Shape<cute::_128, cute::_32>,
        /*StoreAuxD=*/true>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    } else {
      bf16bf16_grouped_gemm_swiglu_impl_sm90<
        a_row_major, b_row_major,
        /*PONGOr2SM*/ false,
        cute::_128, cute::_256, cute::_64,
        cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*RequestedStagesD=*/2,
        /*RequestedMainloopStages=*/4,
        cute::Shape<cute::_128, cute::_32>,
        /*StoreAuxD=*/false>(
          mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out,
          swiglu_alpha, swiglu_limit);
    }
  }
}

void bf16bf16_grouped_mm_swiglu(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor bias,
    const std::optional<at::Tensor>& aux_out_opt,
    at::Tensor& swiglu_out,
    float swiglu_alpha,
    float swiglu_limit,
    bool store_aux) {
  bool a_row_major = mat_a.stride(-1) == 1;
  bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_bf16_grouped_swiglu_on_tile_size<true, true>(
        mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out, swiglu_alpha, swiglu_limit, store_aux);
  } else if (a_row_major && !b_row_major) {
    dispatch_bf16_grouped_swiglu_on_tile_size<true, false>(
        mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out, swiglu_alpha, swiglu_limit, store_aux);
  } else if (!a_row_major && b_row_major) {
    dispatch_bf16_grouped_swiglu_on_tile_size<false, true>(
        mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out, swiglu_alpha, swiglu_limit, store_aux);
  } else {
    dispatch_bf16_grouped_swiglu_on_tile_size<false, false>(
        mat_a, mat_b, offs, bias, aux_out_opt, swiglu_out, swiglu_alpha, swiglu_limit, store_aux);
  }
}
