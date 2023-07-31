#include <cuda_fp16.h>

static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

extern "C" __global__ void __launch_bounds__(32) rotary_mha_decode_kvconst_72_128_64_2048_float16_kernel(half* __restrict__ K_proj, half* __restrict__ O, half* __restrict__ Q_proj, half* __restrict__ V_proj, half* __restrict__ kvbuf, int64_t* __restrict__ kvidx, int64_t* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz) {
  __shared__ half Q_rotary[128];
  __shared__ half K_last_rotary[128];
  float m_now[1];
  float m_prev[1];
  float d_now[1];
  float d_prev[1];
  __shared__ float O_shared[128];
  __shared__ half K_shared[4096];
  float in_thread_X[1];
  float red_buf0[1];
  float red_buf0_1[1];
  __shared__ float A[32];
  float red_buf0_2[1];
  float o[1];
  #pragma unroll
  for (int64_t k_0 = 0; k_0 < 4L; ++k_0) {
    float emb = (((float)past_len[blockIdx.x]) / powf(1.000000e+04f, (((float)(((k_0 & 1L) * 64L) + (threadIdx.x * 2L))) * 7.812500e-03f)));
    float cos = __cosf(emb);
    float sin = __sinf(emb);
    Q_rotary[((k_0 * 32L) + threadIdx.x)] = ((half)((((float)Q_proj[((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x)]) * cos) + (((float)((2L <= k_0) ? Q_proj[(((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x) - 64L)] : (Q_proj[(((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x) + 64L)] * __float2half_rn(-1.000000e+00f)))) * sin)));
    K_last_rotary[((k_0 * 32L) + threadIdx.x)] = ((half)((((float)K_proj[((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x)]) * cos) + (((float)((2L <= k_0) ? K_proj[(((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x) - 64L)] : (K_proj[(((((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (k_0 * 32L)) + threadIdx.x) + 64L)] * __float2half_rn(-1.000000e+00f)))) * sin)));
  }
  __syncthreads();
  *(uint2*)(kvbuf + (((((kvidx[blockIdx.x] * 2415919104L) + (layer_idx * 37748736L)) + (past_len[blockIdx.x] * 9216L)) + (blockIdx.y * 128L)) + (threadIdx.x * 4L))) = *(uint2*)(K_last_rotary + (threadIdx.x * 4L));
  *(uint2*)(kvbuf + ((((((kvidx[blockIdx.x] * 2415919104L) + (layer_idx * 37748736L)) + (past_len[blockIdx.x] * 9216L)) + (blockIdx.y * 128L)) + (threadIdx.x * 4L)) + 18874368L)) = *(uint2*)(V_proj + (((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (threadIdx.x * 4L)));
  m_now[0] = -3.402823e+38f;
  m_prev[0] = -3.402823e+38f;
  d_now[0] = 0.000000e+00f;
  d_prev[0] = 0.000000e+00f;
  #pragma unroll
  for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1) {
    O_shared[((((int64_t)k_0_1) * 32L) + threadIdx.x)] = 0.000000e+00f;
  }
  for (int64_t jo = 0; jo < ((past_len[blockIdx.x] + 31L) >> 5L); ++jo) {
    __syncthreads();
    #pragma unroll
    for (int64_t ji_k_fused_0 = 0; ji_k_fused_0 < 32L; ++ji_k_fused_0) {
      *(uint2*)(K_shared + ((ji_k_fused_0 * 128L) + (threadIdx.x * 4L))) = ((((jo * 32L) + ji_k_fused_0) <= past_len[blockIdx.x]) ? *(uint2*)(kvbuf + ((((((kvidx[blockIdx.x] * 2415919104L) + (layer_idx * 37748736L)) + (jo * 294912L)) + (ji_k_fused_0 * 9216L)) + (blockIdx.y * 128L)) + (threadIdx.x * 4L))) : make_uint2(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f))));
    }
    __syncthreads();
    #pragma unroll
    for (int64_t ji = 0; ji < 32L; ++ji) {
      in_thread_X[0] = 0.000000e+00f;
      #pragma unroll
      for (int k_0_2 = 0; k_0_2 < 4; ++k_0_2) {
        in_thread_X[0] = (in_thread_X[0] + (((((float)Q_rotary[((((int64_t)k_0_2) * 32L) + threadIdx.x)]) * ((float)K_shared[(((ji * 128L) + (((int64_t)k_0_2) * 32L)) + threadIdx.x)])) * 8.838835e-02f) + ((past_len[blockIdx.x] < ((jo * 32L) + ji)) ? -3.402823e+38f : 0.000000e+00f)));
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
    for (int64_t ji_k_fused_0_1 = 0; ji_k_fused_0_1 < 32L; ++ji_k_fused_0_1) {
      *(uint2*)(K_shared + ((ji_k_fused_0_1 * 128L) + (threadIdx.x * 4L))) = ((((jo * 32L) + ji_k_fused_0_1) <= past_len[blockIdx.x]) ? *(uint2*)(kvbuf + (((((((kvidx[blockIdx.x] * 2415919104L) + (layer_idx * 37748736L)) + (jo * 294912L)) + (ji_k_fused_0_1 * 9216L)) + (blockIdx.y * 128L)) + (threadIdx.x * 4L)) + 18874368L)) : make_uint2(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f))));
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
    for (int k_0_3 = 0; k_0_3 < 4; ++k_0_3) {
      #pragma unroll
      for (int ji_1 = 0; ji_1 < 32; ++ji_1) {
        if (ji_1 == 0) {
          o[0] = 0.000000e+00f;
        }
        o[0] = (o[0] + (A[ji_1] * ((float)K_shared[(((((int64_t)ji_1) * 128L) + (((int64_t)k_0_3) * 32L)) + threadIdx.x)])));
      }
      O_shared[((((int64_t)k_0_3) * 32L) + threadIdx.x)] = ((((O_shared[((((int64_t)k_0_3) * 32L) + threadIdx.x)] * d_prev[0]) * __expf((m_prev[0] - m_now[0]))) + o[0]) / d_now[0]);
    }
  }
  __syncthreads();
  uint2 __1;
  float4 v_ = *(float4*)(O_shared + (threadIdx.x * 4L));
  ((half2*)(&(__1.x)))->x = (half)(v_.x);
  ((half2*)(&(__1.x)))->y = (half)(v_.y);
  ((half2*)(&(__1.y)))->x = (half)(v_.z);
  ((half2*)(&(__1.y)))->y = (half)(v_.w);
  *(uint2*)(O + (((blockIdx.x * 9216L) + (blockIdx.y * 128L)) + (threadIdx.x * 4L))) = __1;
}

extern "C" void launch_rotary_mha_decode_kvconst_72_128_64_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz) {
  dim3 grid(B, 72);
  dim3 block(32);
  rotary_mha_decode_kvconst_72_128_64_2048_float16_kernel<<<grid, block>>>((half*)K_proj, (half*)O, (half*)Q_proj, (half*)V_proj, (half*)kvbuf, (int64_t*)kvidx, (int64_t*)past_len, B, layer_idx, nnz);
}