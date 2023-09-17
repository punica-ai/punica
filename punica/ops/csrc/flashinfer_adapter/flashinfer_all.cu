#include "flashinfer_config.h"
#include "flashinfer_impl.cuh"

#define INST_FlashInferBatchDecodeKernel(T, head_dim)                        \
  template void FlashInferBatchDecodeKernel<head_dim, T>(                    \
      T* __restrict__ o, T* __restrict__ q, T* __restrict__ kv_data,         \
      int32_t* __restrict__ kv_indptr, int32_t* __restrict__ kv_indicies,    \
      int32_t* __restrict__ last_page_offset, int num_layers, int layer_idx, \
      int num_heads, int page_size, int batch_size);

FOR_FlashInferBatchDecode_D(INST_FlashInferBatchDecodeKernel, nv_half)
FOR_FlashInferBatchDecode_D(INST_FlashInferBatchDecodeKernel, nv_bfloat16)
