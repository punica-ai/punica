#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
__global__ void precompute_sgmv_args(cutlass::gemm::GemmCoord *all_problems,
                                     T **ptr_x, T **ptr_y, int64_t *ld_x,
                                     int64_t *ld_w, int64_t *ld_y, int32_t *s,
                                     int d_in, int d_out, T *x, T *y) {
  int i = blockIdx.x;
  int m = s[i + 1] - s[i], k = d_in, n = d_out;
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_x[i] = x + s[i] * d_in;
  ptr_y[i] = y + s[i] * d_out;
  ld_x[i] = k;
  ld_w[i] = n;
  ld_y[i] = n;
}

// tmp_d length: num_problems * 52 bytes
template <typename DType>
bool sgmv(DType *y, DType *x, DType **w, int32_t *s, void *tmp_d,
          int num_problems, int d_in, int d_out) {
  using cutlass_t = typename cutlass_dtype<DType>::type;

  char *p = (char *)tmp_d;
  auto ptr_X = (cutlass_t **)p;
  p += num_problems * sizeof(cutlass_t *);
  auto ptr_Y = (cutlass_t **)p;
  p += num_problems * sizeof(cutlass_t *);
  auto ld_X = (int64_t *)p;
  p += num_problems * sizeof(int64_t);
  auto ld_W = (int64_t *)p;
  p += num_problems * sizeof(int64_t);
  auto ld_Y = (int64_t *)p;
  p += num_problems * sizeof(int64_t);
  auto all_problems = (cutlass::gemm::GemmCoord *)p;
  p += num_problems * sizeof(cutlass::gemm::GemmCoord);

  precompute_sgmv_args<cutlass_t>
      <<<num_problems, 1>>>(all_problems, ptr_X, ptr_Y, ld_X, ld_W, ld_Y, s,
                            d_in, d_out, (cutlass_t *)x, (cutlass_t *)y);

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      cutlass_t,                                      // Element A
      cutlass::layout::RowMajor,                      // Layout A
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity A
      cutlass_t,                                      // Element B
      cutlass::layout::RowMajor,                      // Layout B
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity B
      cutlass_t,                                      // Element C&D
      cutlass::layout::RowMajor,                      // Layout C&D
      float,                                          // Element Accumulator
      cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
      cutlass::arch::Sm80,                            // Architecture
      cutlass::gemm::GemmShape<64, 64, 16>,           // Thread Block Shape
      cutlass::gemm::GemmShape<32, 32, 16>,           // Warp Shape
      cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
      LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
      GemmIdentityThreadblockSwizzle<1>,              // Swizzling Operator
      2                                               // Stages
      >::GemmKernel;

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                       epilogue_op, ptr_X, (cutlass_t **)w,
                                       ptr_Y, ptr_Y, ld_X, ld_W, ld_Y, ld_Y);

  GemmGrouped gemm;
  if (gemm.initialize(args) != cutlass::Status::kSuccess) return false;
  if (gemm.run() != cutlass::Status::kSuccess) return false;
  return true;
}
