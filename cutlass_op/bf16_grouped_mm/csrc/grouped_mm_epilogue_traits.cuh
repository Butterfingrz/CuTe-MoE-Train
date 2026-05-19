#pragma once

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

// =============================================================================
// Metafunction: reconstruct a PtrArray CollectiveEpilogue with DelayTmaStore=true.
// The CUTLASS builder forces DelayTmaStore=false for all PtrArray schedules
// (sm90_builder.inl:74), but this is safe to override when ElementC=void because
// ReuseSmemC=false, so the deadlock guard (sm90_epilogue_array_tma_warpspecialized.hpp:812)
// `not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD)` trivially passes.
// Enabling DelayTmaStore defers TMA store of subtile N to subtile N+1's compute phase,
// overlapping the TMA fence/barrier latency with ALU-intensive SwiGLU visit().
// =============================================================================
template <typename CollectiveEpilogue>
struct EnableDelayTmaStore { using type = CollectiveEpilogue; };

template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
  bool DelayTmaStore_, int NumEpiWGs,
  class CtaTileMNK, class EpiTile, class ElementC, class StrideC,
  class ElementD, class StrideD, class FusionCallbacks,
  class CopyOpG2S, class SmemLayoutAtomC, class CopyOpS2R,
  class CopyOpS2G, class SmemLayoutAtomD, class CopyOpR2S,
  class CopyAtomC, class CopyOpR2R>
struct EnableDelayTmaStore<
  cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<
      StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore_, NumEpiWGs>,
    CtaTileMNK, EpiTile, ElementC, StrideC, ElementD, StrideD,
    FusionCallbacks, CopyOpG2S, SmemLayoutAtomC, CopyOpS2R,
    CopyOpS2G, SmemLayoutAtomD, CopyOpR2S, CopyAtomC, CopyOpR2R>
> {
  using type = cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<
      StagesC, StagesD, FragmentSize, ReuseSmemC, true/*DelayTmaStore*/, NumEpiWGs>,
    CtaTileMNK, EpiTile, ElementC, StrideC, ElementD, StrideD,
    FusionCallbacks, CopyOpG2S, SmemLayoutAtomC, CopyOpS2R,
    CopyOpS2G, SmemLayoutAtomD, CopyOpR2S, CopyAtomC, CopyOpR2R>;
};

// =============================================================================
// Metafunction: reconstruct a PtrArray CollectiveEpilogue with a custom StagesD.
// The CUTLASS builder hardcodes StagesD = min(EpiTiles, 2) (sm90_builder.inl:75),
// but for the cooperative tile with 8 EpiTiles and TMA-bound epilogue, StagesD=3
// eliminates pipeline stalls by allowing 2 in-flight TMA batches (UnacquiredStages=2).
// With StagesD=3, TMA(N-3) has 2 subtile compute windows (2x280cy = 560cy) to complete,
// which exceeds the TMA latency (~375cy) -> zero stall in steady state.
// Trade-off: +12 KB smem (8 KB D + 4 KB SwiGLU per stage) may reduce mainloop stages by 1.
// =============================================================================
template <typename CollectiveEpilogue, int NewStagesD>
struct OverrideStagesD { using type = CollectiveEpilogue; };

template <
  int StagesC, int StagesD_, int FragmentSize, bool ReuseSmemC,
  bool DelayTmaStore, int NumEpiWGs,
  class CtaTileMNK, class EpiTile, class ElementC, class StrideC,
  class ElementD, class StrideD, class FusionCallbacks,
  class CopyOpG2S, class SmemLayoutAtomC, class CopyOpS2R,
  class CopyOpS2G, class SmemLayoutAtomD, class CopyOpR2S,
  class CopyAtomC, class CopyOpR2R,
  int NewStagesD>
struct OverrideStagesD<
  cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<
      StagesC, StagesD_, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpiWGs>,
    CtaTileMNK, EpiTile, ElementC, StrideC, ElementD, StrideD,
    FusionCallbacks, CopyOpG2S, SmemLayoutAtomC, CopyOpS2R,
    CopyOpS2G, SmemLayoutAtomD, CopyOpR2S, CopyAtomC, CopyOpR2R>,
  NewStagesD
> {
  using type = cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<
      StagesC, NewStagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumEpiWGs>,
    CtaTileMNK, EpiTile, ElementC, StrideC, ElementD, StrideD,
    FusionCallbacks, CopyOpG2S, SmemLayoutAtomC, CopyOpS2R,
    CopyOpS2G, SmemLayoutAtomD, CopyOpR2S, CopyAtomC, CopyOpR2R>;
};

// =============================================================================
// Metafunction: select mainloop stage count -- fixed or auto-carveout.
// FIXED_STAGES = 0 -> auto mode (StageCountAutoCarveout<CarveoutBytes>)
// FIXED_STAGES > 0 -> fixed mode (StageCount<FIXED_STAGES>)
// =============================================================================
template <int FIXED_STAGES, int CarveoutBytes>
struct MainloopStageCountPolicy {
  using type = cutlass::gemm::collective::StageCount<FIXED_STAGES>;
};

template <int CarveoutBytes>
struct MainloopStageCountPolicy<0, CarveoutBytes> {
  using type = cutlass::gemm::collective::StageCountAutoCarveout<CarveoutBytes>;
};
