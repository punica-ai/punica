#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) bggemv_11008_16_float16_kernel(int64_t* __restrict__ I, half* __restrict__ W, half* __restrict__ X, half* __restrict__ Y, int64_t B, int64_t layer_idx, int64_t num_layers, int64_t num_loras) {
  half Y_rf_local[1];
  __shared__ half red_buf0[256];
  Y_rf_local[0] = __float2half_rn(0.000000e+00f);
  for (int64_t ax0_fused_0 = 0; ax0_fused_0 < 43L; ++ax0_fused_0) {
    Y_rf_local[0] = (Y_rf_local[0] + (X[(((blockIdx.y * 11008L) + (ax0_fused_0 * 256L)) + threadIdx.x)] * W[((((((I[blockIdx.y] * num_layers) * 176128L) + (layer_idx * 176128L)) + (blockIdx.x * 11008L)) + (ax0_fused_0 * 256L)) + threadIdx.x)]));
  }
  __syncthreads();
  ((volatile half*)red_buf0)[threadIdx.x] = Y_rf_local[0];
  __syncthreads();
  if (threadIdx.x < 128L) {
    ((volatile half*)red_buf0)[threadIdx.x] = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 128L)]));
  }
  __syncthreads();
  if (threadIdx.x < 64L) {
    ((volatile half*)red_buf0)[threadIdx.x] = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 64L)]));
  }
  __syncthreads();
  if (threadIdx.x < 32L) {
    ((volatile half*)red_buf0)[threadIdx.x] = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 32L)]));
  }
  __syncthreads();
  if (threadIdx.x < 16L) {
    half w_16_0 = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 16L)]));
    ((volatile half*)red_buf0)[threadIdx.x] = w_16_0;
    half w_8_0 = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 8L)]));
    ((volatile half*)red_buf0)[threadIdx.x] = w_8_0;
    half w_4_0 = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 4L)]));
    ((volatile half*)red_buf0)[threadIdx.x] = w_4_0;
    half w_2_0 = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 2L)]));
    ((volatile half*)red_buf0)[threadIdx.x] = w_2_0;
    half w_1_0 = ((half)(((volatile half*)red_buf0)[threadIdx.x]) + (half)(((volatile half*)red_buf0)[(threadIdx.x + 1L)]));
    ((volatile half*)red_buf0)[threadIdx.x] = w_1_0;
  }
  __syncthreads();
  if (threadIdx.x == 0L) {
    Y[((blockIdx.y * 16L) + blockIdx.x)] = (half)(((volatile half*)red_buf0)[0L]);
  }
}

extern "C" void launch_bggemv_11008_16_float16_kernel(void* __restrict__ I, void* __restrict__ W, void* __restrict__ X, void* __restrict__ Y, int64_t B, int64_t layer_idx, int64_t num_layers, int64_t num_loras) {
  dim3 grid(16, B);
  dim3 block(256);
  bggemv_11008_16_float16_kernel<<<grid, block>>>((int64_t*)I, (half*)W, (half*)X, (half*)Y, B, layer_idx, num_layers, num_loras);
}