// leverage Cutlass 3.x API
#pragma once 

// This is adapated from https://github.com/intel/cutlass-sycl/blob/sycl-develop/examples/sycl/00_bmg_gemm/00_bmg_gemm.cpp.

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

namespace extension_sycl {

using namespace cute;

template<typename DataType, typename OutputType>
void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, OutputType* ptrC) {
    // elements in input matrices.
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = DataType;      // <- data type of elements in input matrix A
  using ElementInputB = DataType;      // <- data type of elements in input matrix B
  using ElementOutput = OutputType;           // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // The 2D block copy operations used for the A and B matrices
  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =                    // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // For Intel BMG, PipelineStages defines how many k-blocks ahead to prefetch from A and B.
  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // This is the 'default' epilogue operation (Linear Combination) which performs everything in:
  // (D = alpha * (A*B) + beta * C)
  // aside from the (A*B), which is handled by the GEMM. See 05_bmg_gemm_with_epilogues for more
  // complex epilogue examples.
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  // FusionCallbacks ties the EpilogueOp to an implementation (based on the dispatch
  // policy/architecture) and defines the epilogue arguments.
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  // GEMM Epilogue - loads & stores C/D matrices, performs epilogue operations & load/stores any
  // auxiliary data required
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N, // The copy atom used to load matrix C
          void, void,
          XE_2D_U32x8x16_ST_N, // The copy atom used to store matrix D
          void, void>;

  // GEMM Mainloop - iteration over blocks in K dimension
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  // Define the whole kernel (mainloop and epilogue)
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>, // Defer global problem shape definition to runtime
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  // The GemmUniversalAdapter wraps the defined GEMM kernel and handles the launch, and e.g.
  // persistent scratch memory if required.
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Gemm gemm_op;

  float alpha = 1.00f;
  float beta = 0.0f;

  //
  // Allocate device memory
  //
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
//   using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
//   StrideD stride_D;

  int L = 1;
  typename Gemm::GemmKernel::ProblemShape problem_mnkl {M, N, K, L};

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
//   stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_mnkl,
      {ptrA, stride_A, ptrB, stride_B},
      {{alpha, beta}, nullptr, stride_C, ptrC, stride_C},
      hw_info
    };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
      std::exit(1);
    }

  gemm_op.initialize(arguments, workspace.get());
  gemm_op.run();
  syclcompat::wait();
}

} // namespace extension_sycl

