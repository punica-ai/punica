#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(32) rotary_mha_decode_96_float16_kernel(half* __restrict__ K_proj, half* __restrict__ O, half* __restrict__ Q_proj, half* __restrict__ V_proj, half* __restrict__ kvbuf, int64_t* __restrict__ kvidx, int64_t* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz) {
  __shared__ half Q_rotary[96];
  __shared__ half K_last_rotary[96];
  float m_now[1];
  float m_prev[1];
  float d_now[1];
  float d_prev[1];
  __shared__ float O_shared[96];
  __shared__ half K_shared[3072];
  float in_thread_X[1];
  float red_buf0[1];
  float red_buf0_1[1];
  __shared__ float A[32];
  float red_buf0_2[1];
  float o[1];
  #pragma unroll
  for (int64_t k_0 = 0; k_0 < 3L; ++k_0) {
    float emb = (((float)past_len[blockIdx.x]) / powf(1.000000e+04f, (((float)((((k_0 * 32L) + threadIdx.x) % 48L) * 2L)) * 1.041667e-02f)));
    float cos = __cosf(emb);
    float sin = __sinf(emb);
    Q_rotary[((k_0 * 32L) + threadIdx.x)] = ((half)((((float)Q_proj[((((blockIdx.y * 96L) + ((blockIdx.x * H) * 96L)) + (k_0 * 32L)) + threadIdx.x)]) * cos) + (((float)((3L <= ((k_0 * 2L) + (threadIdx.x >> 4L))) ? Q_proj[((((((blockIdx.x * H) * 96L) + (blockIdx.y * 96L)) + (k_0 * 32L)) + threadIdx.x) - 48L)] : (Q_proj[(((((blockIdx.y * 96L) + ((blockIdx.x * H) * 96L)) + (k_0 * 32L)) + threadIdx.x) + 48L)] * __float2half_rn(-1.000000e+00f)))) * sin)));
    K_last_rotary[((k_0 * 32L) + threadIdx.x)] = ((half)((((float)K_proj[((((blockIdx.y * 96L) + ((blockIdx.x * H) * 96L)) + (k_0 * 32L)) + threadIdx.x)]) * cos) + (((float)((3L <= ((k_0 * 2L) + (threadIdx.x >> 4L))) ? K_proj[((((((blockIdx.x * H) * 96L) + (blockIdx.y * 96L)) + (k_0 * 32L)) + threadIdx.x) - 48L)] : (K_proj[(((((blockIdx.y * 96L) + ((blockIdx.x * H) * 96L)) + (k_0 * 32L)) + threadIdx.x) + 48L)] * __float2half_rn(-1.000000e+00f)))) * sin)));
  }
  __syncthreads();
  for (int64_t k_2_s = 0; k_2_s < 4L; ++k_2_s) {
    if (threadIdx.x < 24L) {
      kvbuf[((((((((((kvidx[blockIdx.x] * L) * 2L) + (layer_idx * 2L)) * MAXLEN) + past_len[blockIdx.x]) * H) * 96L) + (blockIdx.y * 96L)) + (threadIdx.x * 4L)) + k_2_s)] = K_last_rotary[((threadIdx.x * 4L) + k_2_s)];
    }
  }
  for (int64_t k_2_s_1 = 0; k_2_s_1 < 4L; ++k_2_s_1) {
    if (threadIdx.x < 24L) {
      kvbuf[(((((((((((kvidx[blockIdx.x] * L) * 2L) + (layer_idx * 2L)) + 1L) * MAXLEN) + past_len[blockIdx.x]) * H) * 96L) + (blockIdx.y * 96L)) + (threadIdx.x * 4L)) + k_2_s_1)] = V_proj[((((blockIdx.y * 96L) + ((blockIdx.x * H) * 96L)) + (threadIdx.x * 4L)) + k_2_s_1)];
    }
  }
  m_now[0] = -3.402823e+38f;
  m_prev[0] = -3.402823e+38f;
  d_now[0] = 0.000000e+00f;
  d_prev[0] = 0.000000e+00f;
  #pragma unroll
  for (int k_0_1 = 0; k_0_1 < 3; ++k_0_1) {
    O_shared[((((int64_t)k_0_1) * 32L) + threadIdx.x)] = 0.000000e+00f;
  }
  for (int64_t jo = 0; jo < ((past_len[blockIdx.x] + 31L) >> 5L); ++jo) {
    __syncthreads();
    #pragma unroll
    for (int64_t ji_k_fused_0 = 0; ji_k_fused_0 < 24L; ++ji_k_fused_0) {
      for (int64_t ji_k_fused_2_s = 0; ji_k_fused_2_s < 4L; ++ji_k_fused_2_s) {
        K_shared[(((ji_k_fused_0 * 128L) + (threadIdx.x * 4L)) + ji_k_fused_2_s)] = ((((jo * 32L) + (((ji_k_fused_0 * 4L) + (threadIdx.x >> 3L)) / 3L)) <= past_len[blockIdx.x]) ? kvbuf[(((((((jo * 32L) + (((ji_k_fused_0 * 4L) + (threadIdx.x >> 3L)) / 3L)) + ((((kvidx[blockIdx.x] * L) * 2L) + (layer_idx * 2L)) * MAXLEN)) * H) * 96L) + (blockIdx.y * 96L)) + ((((ji_k_fused_0 * 128L) + (threadIdx.x * 4L)) + ji_k_fused_2_s) % 96L))] : __float2half_rn(0.000000e+00f));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int64_t ji = 0; ji < 32L; ++ji) {
      in_thread_X[0] = 0.000000e+00f;
      #pragma unroll
      for (int k_0_2 = 0; k_0_2 < 3; ++k_0_2) {
        in_thread_X[0] = (in_thread_X[0] + (((((float)Q_rotary[((((int64_t)k_0_2) * 32L) + threadIdx.x)]) * ((float)K_shared[(((ji * 96L) + (((int64_t)k_0_2) * 32L)) + threadIdx.x)])) * 1.020621e-01f) + ((past_len[blockIdx.x] < ((jo * 32L) + ji)) ? -3.402823e+38f : 0.000000e+00f)));
      }
      uint mask[1];
      float t0[1];
      red_buf0[0] = in_thread_X[0];
      mask[0] = __activemask();
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
      red_buf0[0] = (red_buf0[0] + t0[0]);
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
        ((float*)K_last_rotary)[ji] = red_buf0[0];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int64_t ji_k_fused_0_1 = 0; ji_k_fused_0_1 < 24L; ++ji_k_fused_0_1) {
      for (int64_t ji_k_fused_2_s_1 = 0; ji_k_fused_2_s_1 < 4L; ++ji_k_fused_2_s_1) {
        K_shared[(((ji_k_fused_0_1 * 128L) + (threadIdx.x * 4L)) + ji_k_fused_2_s_1)] = ((((jo * 32L) + (((ji_k_fused_0_1 * 4L) + (threadIdx.x >> 3L)) / 3L)) <= past_len[blockIdx.x]) ? kvbuf[(((((((jo * 32L) + (((ji_k_fused_0_1 * 4L) + (threadIdx.x >> 3L)) / 3L)) + (((((kvidx[blockIdx.x] * L) * 2L) + (layer_idx * 2L)) + 1L) * MAXLEN)) * H) * 96L) + (blockIdx.y * 96L)) + ((((ji_k_fused_0_1 * 128L) + (threadIdx.x * 4L)) + ji_k_fused_2_s_1) % 96L))] : __float2half_rn(0.000000e+00f));
      }
    }
    uint mask_1[1];
    float t0_1[1];
    red_buf0_1[0] = ((float*)K_last_rotary)[threadIdx.x];
    mask_1[0] = __activemask();
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    red_buf0_1[0] = __shfl_sync(mask_1[0], red_buf0_1[0], 0, 32);
    m_prev[0] = m_now[0];
    m_now[0] = max(m_prev[0], red_buf0_1[0]);
    A[threadIdx.x] = __expf((((float*)K_last_rotary)[threadIdx.x] - m_now[0]));
    uint mask_2[1];
    float t0_2[1];
    red_buf0_2[0] = A[threadIdx.x];
    mask_2[0] = __activemask();
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 16, 32);
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 8, 32);
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 4, 32);
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
    red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
    red_buf0_2[0] = __shfl_sync(mask_2[0], red_buf0_2[0], 0, 32);
    d_prev[0] = d_now[0];
    d_now[0] = ((d_prev[0] * __expf((m_prev[0] - m_now[0]))) + max(red_buf0_2[0], 1.000000e-05f));
    __syncthreads();
    #pragma unroll
    for (int k_0_3 = 0; k_0_3 < 3; ++k_0_3) {
      #pragma unroll
      for (int ji_1 = 0; ji_1 < 32; ++ji_1) {
        if (ji_1 == 0) {
          o[0] = 0.000000e+00f;
        }
        o[0] = (o[0] + (A[ji_1] * ((float)K_shared[(((((int64_t)ji_1) * 96L) + (((int64_t)k_0_3) * 32L)) + threadIdx.x)])));
      }
      O_shared[((((int64_t)k_0_3) * 32L) + threadIdx.x)] = ((((O_shared[((((int64_t)k_0_3) * 32L) + threadIdx.x)] * d_prev[0]) * __expf((m_prev[0] - m_now[0]))) + o[0]) / d_now[0]);
    }
  }
  __syncthreads();
  for (int64_t k_2_s_2 = 0; k_2_s_2 < 4L; ++k_2_s_2) {
    if (threadIdx.x < 24L) {
      O[(((((blockIdx.x * H) * 96L) + (blockIdx.y * 96L)) + (threadIdx.x * 4L)) + k_2_s_2)] = ((half)O_shared[((threadIdx.x * 4L) + k_2_s_2)]);
    }
  }
}

extern "C" void launch_rotary_mha_decode_96_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz) {
  dim3 grid(B, H);
  dim3 block(32);
  rotary_mha_decode_96_float16_kernel<<<grid, block>>>((half*)K_proj, (half*)O, (half*)Q_proj, (half*)V_proj, (half*)kvbuf, (int64_t*)kvidx, (int64_t*)past_len, B, H, L, MAXLEN, layer_idx, nnz);
}