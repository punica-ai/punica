#pragma once

template <int head_dim, typename T>
void FlashInferBatchDecodeKernel(T* __restrict__ o, T* __restrict__ q,
                                 T* __restrict__ kv_data,
                                 int32_t* __restrict__ kv_indptr,
                                 int32_t* __restrict__ kv_indicies,
                                 int32_t* __restrict__ last_page_offset,
                                 int num_layers, int layer_idx, int num_heads,
                                 int page_size, int batch_size);

// clang-format off

#define FOR_FlashInferBatchDecode_D(f, T) \
    f(T, 64) \
    f(T, 128)

// clang-format on
