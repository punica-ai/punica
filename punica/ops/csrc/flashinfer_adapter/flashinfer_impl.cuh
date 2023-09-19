#pragma once
#include <algorithm>
#include <cmath>

#include "../flashinfer/decode.cuh"
#include "../flashinfer/page.cuh"

template <int head_dim, typename T>
void FlashInferBatchDecodeKernel(T* __restrict__ o, T* __restrict__ q,
                                 T* __restrict__ kv_data,
                                 int32_t* __restrict__ kv_indptr,
                                 int32_t* __restrict__ kv_indicies,
                                 int32_t* __restrict__ last_page_offset,
                                 int num_layers, int layer_idx, int num_heads,
                                 int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  const float rope_scale = 1.f;
  const float rope_theta = 1e4;
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float rope_inv_scale = 1.f / rope_scale;
  const float rope_inv_theta = 1.f / rope_theta;

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  constexpr auto rotary_mode = flashinfer::RotaryMode::kApplyRotary;
  dim3 nblks(paged_kv.batch_size, paged_kv.num_heads);
  dim3 nthrs(bdx, bdy);
  flashinfer::BatchDecodeWithPagedKVCacheKernel<rotary_mode, head_dim, vec_size,
                                                bdx, bdy><<<nblks, nthrs>>>(
      q, paged_kv, o, sm_scale, rope_inv_scale, rope_inv_theta);
}

template <int head_dim, typename T>
void FlashInferInitKvKernel(T* __restrict__ kv_data,
                            int32_t* __restrict__ kv_indptr,
                            int32_t* __restrict__ kv_indicies,
                            int32_t* __restrict__ last_page_offset,
                            T* __restrict__ key, T* __restrict__ value,
                            int32_t* __restrict__ seqlen_indptr, int num_layers,
                            int layer_idx, int num_heads, int page_size,
                            int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy, T,
                                              int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value, seqlen_indptr);
}

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T* __restrict__ kv_data,
                              int32_t* __restrict__ kv_indptr,
                              int32_t* __restrict__ kv_indicies,
                              int32_t* __restrict__ last_page_offset,
                              T* __restrict__ key, T* __restrict__ value,
                              int num_layers, int layer_idx, int num_heads,
                              int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy, T,
                                             int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value);
}
