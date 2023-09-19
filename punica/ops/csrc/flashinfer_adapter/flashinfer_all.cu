#include "flashinfer_config.h"
#include "flashinfer_impl.cuh"

#define INST_FlashInferBatchDecodeKernel(head_dim, T)                        \
  template void FlashInferBatchDecodeKernel<head_dim, T>(                    \
      T* __restrict__ o, T* __restrict__ q, T* __restrict__ kv_data,         \
      int32_t* __restrict__ kv_indptr, int32_t* __restrict__ kv_indicies,    \
      int32_t* __restrict__ last_page_offset, int num_layers, int layer_idx, \
      int num_heads, int page_size, int batch_size);

FOR_FlashInferBatchDecode_D(INST_FlashInferBatchDecodeKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferBatchDecodeKernel, nv_bfloat16);

#define INST_FlashInferInitKvKernel(head_dim, T)                   \
  template void FlashInferInitKvKernel<head_dim, T>(               \
      T* __restrict__ kv_data, int32_t* __restrict__ kv_indptr,    \
      int32_t* __restrict__ kv_indicies,                           \
      int32_t* __restrict__ last_page_offset, T* __restrict__ key, \
      T* __restrict__ value, int32_t* __restrict__ seqlen_indptr,  \
      int num_layers, int layer_idx, int num_heads, int page_size, \
      int batch_size);

FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_bfloat16);

#define INST_FlashInferAppendKvKernel(head_dim, T)                         \
  template void FlashInferAppendKvKernel<head_dim, T>(                     \
      T* __restrict__ kv_data, int32_t* __restrict__ kv_indptr,            \
      int32_t* __restrict__ kv_indicies,                                   \
      int32_t* __restrict__ last_page_offset, T* __restrict__ key,         \
      T* __restrict__ value, int num_layers, int layer_idx, int num_heads, \
      int page_size, int batch_size);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_bfloat16);
