#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(16) bggemv_16_4096_float16_kernel(int64_t* __restrict__ I, half* __restrict__ W, half* __restrict__ X, half* __restrict__ Y, int64_t B, int64_t layer_idx, int64_t num_layers, int64_t num_loras) {
  half Y_rf_local[1];
  half red_buf0[1];
  Y_rf_local[0] = __float2half_rn(0.000000e+00f);
  Y_rf_local[0] = (Y_rf_local[0] + (X[((blockIdx.y * 16L) + threadIdx.x)] * W[(((((I[blockIdx.y] * num_layers) * 65536L) + (layer_idx * 65536L)) + (blockIdx.x * 16L)) + threadIdx.x)]));
  uint mask[1];
  half t0[1];
  red_buf0[0] = Y_rf_local[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  if (threadIdx.x == 0L) {
    Y[((blockIdx.y * 4096L) + blockIdx.x)] = red_buf0[0];
  }
}

extern "C" void launch_bggemv_16_4096_float16_kernel(void* __restrict__ I, void* __restrict__ W, void* __restrict__ X, void* __restrict__ Y, int64_t B, int64_t layer_idx, int64_t num_layers, int64_t num_loras) {
  dim3 grid(4096, B);
  dim3 block(16);
  bggemv_16_4096_float16_kernel<<<grid, block>>>((int64_t*)I, (half*)W, (half*)X, (half*)Y, B, layer_idx, num_layers, num_loras);
}