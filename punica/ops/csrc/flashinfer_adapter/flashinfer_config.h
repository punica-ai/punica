#pragma once

template <int head_dim, typename T>
void FlashInferBatchDecodeKernel(T* __restrict__ o, T* __restrict__ q,
                                 T* __restrict__ kv_data,
                                 int32_t* __restrict__ kv_indptr,
                                 int32_t* __restrict__ kv_indicies,
                                 int32_t* __restrict__ last_page_offset,
                                 int num_layers, int layer_idx, int num_heads,
                                 int page_size, int batch_size);

template <int head_dim, typename T>
void FlashInferInitKvKernel(T* __restrict__ kv_data,
                            int32_t* __restrict__ kv_indptr,
                            int32_t* __restrict__ kv_indicies,
                            int32_t* __restrict__ last_page_offset,
                            T* __restrict__ key, T* __restrict__ value,
                            int32_t* __restrict__ seqlen_indptr, int num_layers,
                            int layer_idx, int num_heads, int page_size,
                            int batch_size);

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T* __restrict__ kv_data,
                              int32_t* __restrict__ kv_indptr,
                              int32_t* __restrict__ kv_indicies,
                              int32_t* __restrict__ last_page_offset,
                              T* __restrict__ key, T* __restrict__ value,
                              int num_layers, int layer_idx, int num_heads,
                              int page_size, int batch_size);

// clang-format off

#define FOR_FlashInferBatchDecode_D(f, ...) \
    f(64, __VA_ARGS__) \
    f(128, __VA_ARGS__)

// clang-format on
