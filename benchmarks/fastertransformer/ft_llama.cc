// Adapted from
// https://github.com/void-main/FasterTransformer/blob/e770ddf2bc66217034b6e9e3b0c3256ebf1c1b40/examples/cpp/llama/llama_example.cc
/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/utils/cuda_utils.h"

#include "ft_llama.h"

using namespace fastertransformer;

struct ModelConfig {
    size_t num_heads;
    size_t head_dim;
    size_t inter_size;
    size_t num_layers;
    int    device_id;
};

template<typename T>
class FtLlamaImpl {
public:
    FtLlamaImpl(ModelConfig config);
    ~FtLlamaImpl();
    FtLlamaImpl(const FtLlamaImpl&) = delete;
    FtLlamaImpl& operator=(const FtLlamaImpl&) = delete;

    void
    forward(const std::vector<std::vector<int>>& input_ids, size_t request_output_len, std::function<void()> callback);

private:
    const int    start_id                   = 0;
    const int    end_id                     = 1;
    const size_t beam_width                 = 1;
    const uint   top_k                      = 1;
    const float  top_p                      = 0.0;
    const float  temperature                = 0.0;
    const float  repetition_penalty         = 1.0;
    const float  presence_penalty           = 0.0;
    const float  len_penalty                = 0.0;
    const float  beam_search_diversity_rate = 0.0;
    const int    min_length                 = 0;

    const unsigned long long random_seed = 0xabcdabcd987;

    NcclParam                       tensor_para;
    NcclParam                       pipeline_para;
    cudaStream_t                    stream;
    cublasHandle_t                  cublas_handle;
    cublasLtHandle_t                cublaslt_handle;
    cublasAlgoMap*                  cublas_algo_map;
    Allocator<AllocatorType::CUDA>* allocator;
    std::mutex*                     cublas_wrapper_mutex;
    cublasMMWrapper*                cublas_wrapper;
    Llama<T>*                       gpt;
    LlamaWeight<T>*                 gpt_weights;
};

struct CallbackContext {
    std::function<void()> callback;
};

FtLlama::FtLlama(
    size_t num_heads, size_t head_dim, size_t inter_size, size_t num_layers, const char* data_type, int device_id)
{
    ModelConfig config = {
        .num_heads  = num_heads,
        .head_dim   = head_dim,
        .inter_size = inter_size,
        .num_layers = num_layers,
        .device_id  = device_id,
    };
    if (strcmp(data_type, "float32") == 0) {
        dtype_ = DataType::FP32;
        impl_  = new FtLlamaImpl<float>(config);
    }
    else if (strcmp(data_type, "float16") == 0) {
        dtype_ = DataType::FP16;
        impl_  = new FtLlamaImpl<half>(config);
    }
#ifdef ENABLE_BF16
    else if (strcmp(data_type, "bfloat16") == 0) {
        dtype_ = DataType::BF16;
        impl_  = new FtLlamaImpl<__nv_bfloat16>(config);
    }
#endif
}

FtLlama::~FtLlama()
{
    if (dtype_ == DataType::FP32) {
        delete static_cast<FtLlamaImpl<float>*>(impl_);
    }
    else if (dtype_ == DataType::FP16) {
        delete static_cast<FtLlamaImpl<half>*>(impl_);
    }
#ifdef ENABLE_BF16
    else if (dtype_ == DataType::BF16) {
        delete static_cast<FtLlamaImpl<__nv_bfloat16>*>(impl_);
    }
#endif
}

void FtLlama::forward(const std::vector<std::vector<int>>& input_ids,
                      size_t                               request_output_len,
                      std::function<void()>                callback)
{
    if (dtype_ == DataType::FP32) {
        static_cast<FtLlamaImpl<float>*>(impl_)->forward(input_ids, request_output_len, std::move(callback));
    }
    else if (dtype_ == DataType::FP16) {
        static_cast<FtLlamaImpl<half>*>(impl_)->forward(input_ids, request_output_len, std::move(callback));
    }
#ifdef ENABLE_BF16
    else if (dtype_ == DataType::BF16) {
        static_cast<FtLlamaImpl<__nv_bfloat16>*>(impl_)->forward(input_ids, request_output_len, std::move(callback));
    }
#endif
}

template<typename T>
FtLlamaImpl<T>::FtLlamaImpl(ModelConfig config)
{
    int tensor_para_size   = 1;
    int pipeline_para_size = 1;
    int int8_mode          = 0;

    const size_t head_num             = config.num_heads;
    const size_t size_per_head        = config.head_dim;
    const size_t vocab_size           = 32000;
    const size_t decoder_layers       = config.num_layers;
    const size_t rotary_embedding_dim = 128;
    const float  layernorm_eps        = 1e-5;
    const size_t hidden_units         = head_num * size_per_head;
    const size_t inter_size           = config.inter_size;

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);
    FT_CHECK_WITH_INFO(repetition_penalty == 1.0f || presence_penalty == 0.0f,
                       fmtstr("Found ambiguous parameters repetition_penalty "
                              "(%f) and presence_penalty (%f) "
                              "which are mutually exclusive. Please remove one "
                              "of repetition_penalty or presence_penalty "
                              "or set to a default value.",
                              repetition_penalty,
                              presence_penalty));

    // Prepare the parallelism parameters
    int device;
    check_cuda_error(cudaSetDevice(config.device_id));
    check_cuda_error(cudaGetDevice(&device));
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    const int layers_per_group = decoder_layers / pipeline_para_size;

    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublas_algo_map = new cublasAlgoMap("gemm_config.in");
    allocator       = new Allocator<AllocatorType::CUDA>(getDevice());

    cublas_wrapper_mutex = new std::mutex();
    cublas_wrapper =
        new cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper->setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    const bool use_gptj_residual = false;

    gpt_weights = new LlamaWeight<T>(hidden_units,
                                     inter_size,
                                     vocab_size,
                                     decoder_layers,
                                     0,  // max_seq_len, deprecated
                                     tensor_para.world_size_,
                                     tensor_para.rank_,
                                     pipeline_para.world_size_,
                                     pipeline_para.rank_,
                                     use_gptj_residual,
                                     int8_mode);

    AttentionType attention_type = getAttentionType<T>(size_per_head,
                                                       getSMVersion(),
                                                       true,   // remove_padding
                                                       0,      // gpt supports any-seq-length fmha
                                                       true,   // is_fuse
                                                       false,  // with_relative_position_bias
                                                       true);  // causal_mask

    setenv("LLAMA_STREAM_CB_STEP", "1", 1);
    gpt = new Llama<T>(head_num,
                       size_per_head,
                       inter_size,
                       decoder_layers,
                       vocab_size,
                       rotary_embedding_dim,
                       layernorm_eps,
                       start_id,
                       end_id,
                       end_id + 1,  // prompt_learning_start_id
                       PromptLearningType::no_prompt,
                       use_gptj_residual,
                       0.0f,
                       top_k,
                       top_p,
                       random_seed,
                       temperature,
                       len_penalty,
                       repetition_penalty,
                       stream,
                       cublas_wrapper,
                       allocator,
                       false,
                       &prop,
                       attention_type,
                       int8_mode,
                       nullptr,
                       0,
                       1.0f);
    unsetenv("LLAMA_STREAM_CB_STEP");
}

void flat_pad_input_ids(const std::vector<std::vector<int>>& input_ids,
                        std::vector<int>&                    v_start_lengths,
                        std::vector<int>&                    v_start_ids,
                        size_t&                              max_input_len)
{
    max_input_len = 0;
    for (const auto& ids : input_ids) {
        max_input_len = std::max(max_input_len, ids.size());
    }

    v_start_ids.clear();
    v_start_ids.reserve(input_ids.size() * max_input_len);
    v_start_lengths.clear();
    v_start_lengths.reserve(input_ids.size());
    for (const auto& ids : input_ids) {
        std::copy(ids.begin(), ids.end(), std::back_inserter(v_start_ids));
        std::fill_n(std::back_inserter(v_start_ids), max_input_len - ids.size(), 0);
        v_start_lengths.push_back(ids.size());
    }
}

template<typename T>
void FtLlamaImpl<T>::forward(const std::vector<std::vector<int>>& input_ids,
                             size_t                               request_output_len,
                             std::function<void()>                callback)
{
    CallbackContext ctx = {std::move(callback)};
    if (ctx.callback) {
        gpt->registerCallback(
            [](std::unordered_map<std::string, Tensor>* _out, void* cb_ctx) {
                auto* ctx = static_cast<CallbackContext*>(cb_ctx);
                ctx->callback();
            },
            &ctx);
    }

    // Read ids of request from args.
    size_t           max_input_len;
    size_t           total_outout_len;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    flat_pad_input_ids(input_ids, v_start_lengths, v_start_ids, max_input_len);
    size_t request_batch_size = v_start_lengths.size();

    int* d_input_ids;
    int* d_input_lengths;
    FT_CHECK(max_input_len > 0);
    {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size);
    }
    std::vector<int> start_ids(request_batch_size, start_id);
    std::vector<int> end_ids(request_batch_size, end_id);

    const int total_output_len = max_input_len + request_output_len;

    int* d_output_ids;
    int* d_sequence_lengths;

    deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);

    std::vector<uint32_t>                   output_seq_len(request_batch_size, total_output_len);
    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)max_input_len}, d_input_ids}},
        {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
        // NOTE: if you need prefix prompts, remember to add
        // prefix_prompt_task_ids here
        // {"prompt_learning_task_name_ids", Tensor{MEMORY_CPU, TYPE_INT32,
        // std::vector<size_t>{request_batch_size},
        // prefix_prompt_task_ids.data()}},
        {"output_seq_len",
         Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}},
        {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}},
        {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}},
        {"min_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &min_length}},
        {"start_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, start_ids.data()}},
        {"end_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, end_ids.data()}}};

    if (repetition_penalty != 1.0f) {
        input_tensors.insert(
            {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    }
    if (presence_penalty != 0.0f) {
        input_tensors.insert(
            {"presence_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &presence_penalty}});
    }

    if (top_k == 0 && top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        input_tensors.insert({"beam_search_diversity_rate",
                              Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    }
    else {
        input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
        if (top_p != 0.0f) {
            input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k != 0) {
            input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
        }
    }

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                d_output_ids}},
        {"sequence_length",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}},
        {"output_log_probs",
         Tensor{MEMORY_GPU,
                TYPE_FP32,
                std::vector<size_t>{request_batch_size, (size_t)request_output_len, beam_width},
                nullptr}}};

    gpt->forward(&output_tensors, &input_tensors, gpt_weights);
    if (ctx.callback) {
        ctx.callback();
    }

    if (d_input_ids != nullptr) {
        cudaFree(d_input_ids);
    }
    if (d_input_lengths != nullptr) {
        cudaFree(d_input_lengths);
    }
    if (d_output_ids != nullptr) {
        deviceFree(d_output_ids);
    }
    if (d_sequence_lengths != nullptr) {
        deviceFree(d_sequence_lengths);
    }
    gpt->unRegisterCallback();
}

template<typename T>
FtLlamaImpl<T>::~FtLlamaImpl()
{
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
}
