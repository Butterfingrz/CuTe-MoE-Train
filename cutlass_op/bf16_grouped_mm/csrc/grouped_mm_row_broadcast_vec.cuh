#pragma once

// Sm90RowBroadcastVec: CUTLASS Sm90RowBroadcast variant for grouped GEMM bias fusion.
//
// Uses cp.async for asynchronous gmem→smem bias loading in the consumer
//
// Drop-in replacement: same template signature, SharedStorage, and Arguments.

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/helper_macros.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/copy_sm80.hpp"
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

namespace cutlass::epilogue::fusion {

using namespace cute;

template <
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true
>
struct Sm90RowBroadcastVec {
  using StrideMNL = StrideMNL_;
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrRowType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>;
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast);
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast);

  struct SharedStorage {
    array_aligned<ElementInput, size<1>(CtaTileShapeMNK{})> smem;
  };

  struct Arguments {
    PtrRowType ptr_row = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90RowBroadcastVec() { }

  CUTLASS_HOST_DEVICE
  Sm90RowBroadcastVec(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    if (EnableNullptr && params.ptr_row == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_row[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class SR_STensor, class SR_RTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        SR_STensor tSR_sRow_, SR_RTensor tSR_rRow_,
        ElementInput const* gmem_row_base, ElementInput* smem_base,
        int thread_idx, int thread_count, int valid_n,
        Params const& params_)
      : tSR_sRow(tSR_sRow_)
      , tSR_rRow(tSR_rRow_)
      , gmem_row_base_(gmem_row_base)
      , smem_base_(smem_base)
      , thread_idx_(thread_idx)
      , thread_count_(thread_count)
      , valid_n_(valid_n)
      , params(params_) {
    }

    SR_STensor tSR_sRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    SR_RTensor tSR_rRow;                                                         // (CPY,CPY_M,CPY_N)

    ElementInput const* gmem_row_base_;
    ElementInput* smem_base_;
    int thread_idx_;
    int thread_count_;
    int valid_n_;
    Params const& params;

    // Asynchronous gmem → smem copy using cp.async
    // Loads CTA_N bias elements in 16-byte chunks with zero-fill predication.
    // Stays within Stages=0 consumer-warp model
    CUTLASS_DEVICE void
    begin() {
      constexpr int cta_n = size<1>(CtaTileShapeMNK{});
      bool is_nullptr = EnableNullptr && params.ptr_row == nullptr;

      if (is_nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = thread_idx_; i < cta_n; i += thread_count_) {
          smem_base_[i] = params.null_default;
        }
        return;
      }

      // cp.async 16-byte vectorized async gmem → smem copies
      using CpType = uint_bit_t<128>;
      constexpr int CP_ELEMS = 16 / static_cast<int>(sizeof(ElementInput));
      constexpr int TOTAL_CPS = cta_n / CP_ELEMS;
      static_assert(cta_n % CP_ELEMS == 0, "CTA_N must be a multiple of cp.async element count");

      CUTLASS_PRAGMA_UNROLL
      for (int cp = thread_idx_; cp < TOTAL_CPS; cp += thread_count_) {
        int elem_idx = cp * CP_ELEMS;
        bool pred = (elem_idx + CP_ELEMS <= valid_n_);
        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CpType>::copy(
            reinterpret_cast<CpType const&>(gmem_row_base_[elem_idx]),
            reinterpret_cast<CpType      &>(smem_base_[elem_idx]),
            pred);
      }
      cp_async_fence();
      cp_async_wait<0>();

      // Fixup OOB smem slots when null_default != 0 (ZFILL wrote zeros)
      if (params.null_default != ElementInput(0)) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = thread_idx_; i < cta_n; i += thread_count_) {
          if (i >= valid_n_) {
            smem_base_[i] = params.null_default;
          }
        }
      }
    }

    CUTLASS_DEVICE bool
    begin_sync_needed() const {
      return true;
    }

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      if (epi_m == 0) { // Assumes M-major subtile loop
        auto tSR_sRow_flt = filter_zeros(tSR_sRow(_,_,_,epi_m,epi_n));
        auto tSR_rRow_flt = make_tensor_like<ElementInput>(tSR_sRow_flt);
        copy_aligned(tSR_sRow_flt, tSR_rRow_flt);

        constexpr int FrgSize = size(tSR_rRow_flt);
        using FrgInput = Array<ElementInput, FrgSize>;
        using FrgCompute = Array<ElementCompute, FrgSize>;
        using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FrgSize>;

        auto tSR_rRow_input_frg = recast<FrgInput>(coalesce(tSR_rRow_flt));
        auto tSR_rRow_compute_frg = recast<FrgCompute>(filter(tSR_rRow));
        ConvertInput convert_input{};

        tSR_rRow_compute_frg(_0{}) = convert_input(tSR_rRow_input_frg(_0{}));
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = tSR_rRow(epi_v * FragmentSize + i);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using ThreadCount = decltype(size(args.tiled_copy));

    // Resolve bias gmem pointer for this group
    ElementInput const* ptr_row = nullptr;
    if constexpr(IsArrayOfPointers) {
      if (!(EnableNullptr && params.ptr_row == nullptr)) {
        ptr_row = params.ptr_row[l];
      }
    } else {
      ptr_row = params.ptr_row;
    }

    // Compute valid column count and gmem base for cp.async
    constexpr int cta_n = size<1>(CtaTileShapeMNK{});
    int tile_n_start = static_cast<int>(n) * cta_n;
    int valid_n = static_cast<int>(N) - tile_n_start;
    valid_n = (valid_n > cta_n) ? cta_n : valid_n;
    valid_n = (valid_n < 0) ? 0 : valid_n;

    ElementInput const* gmem_row_base = nullptr;
    if (ptr_row != nullptr) {
      gmem_row_base = ptr_row + tile_n_start;
    }

    // S2R: construct smem tensor and partition for epilogue
    auto sRow = make_tensor(make_smem_ptr(smem),
        make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)
    auto tSR_sRow = sm90_partition_for_epilogue<ReferenceSrc>(sRow, args.epi_tile, args.tiled_copy, args.thread_idx);
    auto tSR_rRow = make_tensor_like<ElementCompute>(take<0,3>(tSR_sRow));                        // (CPY,CPY_M,CPY_N)

    return ConsumerStoreCallbacks<decltype(tSR_sRow), decltype(tSR_rRow)>(
      tSR_sRow,
      tSR_rRow,
      gmem_row_base,
      smem,
      args.thread_idx,
      static_cast<int>(ThreadCount::value),
      valid_n,
      params);
  }
};

} // namespace cutlass::epilogue::fusion