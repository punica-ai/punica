#include "cutlass/cutlass.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/util/host_tensor.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/util/device_memory.h"

template <typename T> struct cutlass_dtype {
  using type = T;
};

template <> struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <> struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
void sgmv_internal(cutlass::gemm::GemmCoord *all_problems, T **ptr_X, T **ptr_W,
                   T **ptr_Y, int64_t *ld_X, int64_t *ld_W, int64_t *ld_Y,
                   int batch_size) {
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      T,                                            // Element A
      cutlass::layout::RowMajor,                    // Layout A
      cutlass::ComplexTransform::kNone,             //
      8,                                            // Granularity A
      T,                                            // Element B
      cutlass::layout::RowMajor,                    // Layout B
      cutlass::ComplexTransform::kNone,             //
      8,                                            // Granularity B
      T,                                            // Element C&D
      cutlass::layout::RowMajor,                    // Layout C&D
      float,                                        // Element Accumulator
      cutlass::arch::OpClassTensorOp,               // Operator Class Tag
      cutlass::arch::Sm80,                          // Architecture
      cutlass::gemm::GemmShape<64, 64, 16>,         // Thread Block Shape
      cutlass::gemm::GemmShape<32, 32, 16>,         // Warp Shape
      cutlass::gemm::GemmShape<16, 8, 8>,           // Instruction Shape
      cutlass::epilogue::thread::LinearCombination< // Epilogue
          T, 8, float, float>,                      //
      cutlass::gemm::threadblock::                  // Swizzling Operator
      GemmIdentityThreadblockSwizzle<1>,            //
      2                                             // Stages
      >::GemmKernel;

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(all_problems, batch_size, 512,
                                       epilogue_op, ptr_X, ptr_W, ptr_Y, ptr_Y,
                                       ld_X, ld_W, ld_Y, ld_Y);

  GemmGrouped gemm;
  auto cutlass_status = gemm.initialize(args);
  if (cutlass_status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize GemmGrouped" << std::endl;
    return;
  }
  cutlass_status = gemm.run();
  if (cutlass_status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to run GemmGrouped" << std::endl;
    return;
  }
}

template <typename T>
__global__ void precompute_args(cutlass::gemm::GemmCoord *all_problems,
                                T **ptr_x, T **ptr_w, T **ptr_y, int64_t *ld_x,
                                int64_t *ld_w, int64_t *ld_y, int32_t *s,
                                int d_in, int d_out, T *x, T *w, T *y, int *I) {
  int i = blockIdx.x;
  int m = s[i + 1] - s[i], k = d_in, n = d_out;
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_x[i] = x + s[i] * d_in;
  ptr_w[i] = w + I[i] * d_in * d_out;
  ptr_y[i] = y + s[i] * d_out;
  ld_x[i] = k;
  ld_w[i] = n;
  ld_y[i] = n;
}

template <typename DType>
void sgmv(DType *y_gpu, DType *x_gpu, DType *w_gpu, int *s_gpu, int *I_gpu,
          int batch_size, int d_in, int d_out) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  cutlass::DeviceAllocation<cutlass_t *> ptr_X, ptr_W, ptr_Y;
  cutlass::DeviceAllocation<int64_t> ld_X, ld_W, ld_Y;
  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems;
  ptr_X.reset(batch_size);
  ptr_W.reset(batch_size);
  ptr_Y.reset(batch_size);
  ld_X.reset(batch_size);
  ld_W.reset(batch_size);
  ld_Y.reset(batch_size);
  all_problems.reset(batch_size);
  precompute_args<cutlass_t><<<batch_size, 1>>>(
      all_problems.get(), ptr_X.get(), ptr_W.get(), ptr_Y.get(), ld_X.get(),
      ld_W.get(), ld_Y.get(), s_gpu, d_in, d_out, (cutlass_t *)x_gpu,
      (cutlass_t *)w_gpu, (cutlass_t *)y_gpu, I_gpu);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::cerr << "Failed to precompute args" << std::endl;
    return;
  }

  sgmv_internal(all_problems.get(), ptr_X.get(), ptr_W.get(), ptr_Y.get(),
                ld_X.get(), ld_W.get(), ld_Y.get(), batch_size);
}
