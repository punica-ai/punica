#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../flashinfer/decode.cuh"
#include "../flashinfer/page.cuh"
#include "flashinfer_config.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

flashinfer::paged_kv_t<nv_half, int32_t> hack_paged_kv(0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
uint32_t hack_tmp_size=0;

template <typename T>
inline T *alloc_from_buf(void **buf, int n) {
  auto *p = (T *)*buf;
  *buf = (void *)(p + n);
  return p;
}

template <typename T>
void FlashInferBatchDecodeKernel(T* o, T* q, T* kv_data, int32_t* kv_indptr,
                                 int32_t* kv_indicies,
                                 int32_t* last_page_offset, void* tmp, void* tmp2, int head_dim,
                                 int num_layers, int layer_idx,
                                 int num_qo_heads, int num_kv_heads,
                                 int page_size, int batch_size) {
  auto rotary_mode = flashinfer::RotaryMode::kLlama;

  if (layer_idx == 0) {
    hack_paged_kv = flashinfer::paged_kv_t<T, int32_t>(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

    uint32_t tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size;
    flashinfer::BatchDecodeWithPagedKVCacheWorkEstimation<T, T>(
            tmp_size, max_grid_size, max_num_pages_per_batch, new_batch_size,
            hack_paged_kv, num_qo_heads, rotary_mode);
    hack_tmp_size = tmp_size;
    if (tmp_size != 0) {
      hack_paged_kv.batch_size = new_batch_size;
      hack_paged_kv.indptr = alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);
      hack_paged_kv.last_page_offset = alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);
      hack_paged_kv.cooperative_indptr = alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);
      hack_paged_kv.batch_idx_map = alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);
      hack_paged_kv.chunk_start = alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);
      hack_paged_kv.seq_lens_before_split =alloc_from_buf<int32_t>(&tmp, new_batch_size + 1);

      thrust::device_vector<int32_t> kv_indptr_d(kv_indptr, kv_indptr+batch_size+1);
      thrust::host_vector<int32_t> kv_indptr_host(kv_indptr_d);

      thrust::device_vector<int32_t> kv_last_page_offset_d(last_page_offset, last_page_offset+batch_size);
      thrust::host_vector<int32_t> kv_last_page_offset_host(kv_last_page_offset_d);

      flashinfer::SplitPagedKVCache(batch_size, kv_indptr_host.data(),
                                    kv_last_page_offset_host.data(),
                                    max_num_pages_per_batch, &hack_paged_kv);
    }
  }

  flashinfer::BatchDecodeWithPagedKVCache(q, hack_paged_kv, o,
    (float*)(hack_tmp_size == 0 ? nullptr : tmp2),
    num_qo_heads,
                                          rotary_mode);
}

template <int head_dim, typename T>
void FlashInferInitKvKernel(T* kv_data, int32_t* kv_indptr,
                            int32_t* kv_indicies, int32_t* last_page_offset,
                            T* key, T* value, int32_t* seqlen_indptr,
                            int num_layers, int layer_idx, int num_kv_heads,
                            int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 1;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy, T,
                                              int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value, seqlen_indptr);
}

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T* kv_data, int32_t* kv_indptr,
                              int32_t* kv_indicies, int32_t* last_page_offset,
                              T* key, T* value, int num_layers, int layer_idx,
                              int num_kv_heads, int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 1;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy, T,
                                             int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value);
}

#define INST_FlashInferBatchDecodeKernel(T)                                    \
  template void FlashInferBatchDecodeKernel<T>(                                \
      T * o, T * q, T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,   \
      int32_t * last_page_offset,void* tmp, void* tmp2,  int head_dim, int num_layers, int layer_idx, \
      int num_qo_heads, int num_kv_heads, int page_size, int batch_size);

INST_FlashInferBatchDecodeKernel(nv_half);

#define INST_FlashInferInitKvKernel(head_dim, T)                               \
  template void FlashInferInitKvKernel<head_dim, T>(                           \
      T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,                 \
      int32_t * last_page_offset, T * key, T * value, int32_t * seqlen_indptr, \
      int num_layers, int layer_idx, int num_kv_heads, int page_size,          \
      int batch_size);

FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_bfloat16);

#define INST_FlashInferAppendKvKernel(head_dim, T)                    \
  template void FlashInferAppendKvKernel<head_dim, T>(                \
      T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,        \
      int32_t * last_page_offset, T * key, T * value, int num_layers, \
      int layer_idx, int num_kv_heads, int page_size, int batch_size);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_bfloat16);
