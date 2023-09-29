#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "bgmv/bgmv_config.h"
#include "flashinfer_adapter/flashinfer_config.h"
#include "gen/punica_ops.cc.inc"
#include "rms_norm/rms_norm.h"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(scalar_type, ...) \
  [&]() -> bool {                          \
    switch (scalar_type) {                 \
      __VA_ARGS__                          \
      default:                             \
        return false;                      \
    }                                      \
  }()

#define _DISPATCH_CASE(enum_type, c_type_, ...) \
  case enum_type: {                             \
    using c_type = c_type_;                     \
    return __VA_ARGS__();                       \
  }

#define _DISPATCH_CASES(...)                                 \
  _DISPATCH_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_CASES(__VA_ARGS__))

//====== rotary_mha_decode ======

template <typename F>
inline void rotary_mha_decode_kvconst(F kernel, torch::Tensor q_proj,
                                      torch::Tensor k_proj,
                                      torch::Tensor v_proj, torch::Tensor o,
                                      torch::Tensor past_len,
                                      torch::Tensor kvbuf, torch::Tensor kvidx,
                                      int64_t layer_idx) {
  int64_t B = q_proj.size(0);
  int64_t nnz = kvbuf.size(0);
  kernel(k_proj.data_ptr(), o.data_ptr(), q_proj.data_ptr(), v_proj.data_ptr(),
         kvbuf.data_ptr(), kvidx.data_ptr(), past_len.data_ptr(), B, layer_idx,
         nnz);
}

#define DEFINE_rotary_mha_decode_kvconst(name)                                \
  void name(torch::Tensor q_proj, torch::Tensor k_proj, torch::Tensor v_proj, \
            torch::Tensor o, torch::Tensor past_len, torch::Tensor kvbuf,     \
            torch::Tensor kvidx, int64_t layer_idx) {                         \
    rotary_mha_decode_kvconst(launch_##name##_kernel, q_proj, k_proj, v_proj, \
                              o, past_len, kvbuf, kvidx, layer_idx);          \
  }

template <typename F>
inline void rotary_mha_decode(F kernel, torch::Tensor q_proj,
                              torch::Tensor k_proj, torch::Tensor v_proj,
                              torch::Tensor o, torch::Tensor past_len,
                              torch::Tensor kvbuf, torch::Tensor kvidx,
                              int64_t layer_idx) {
  int64_t B = q_proj.size(0);
  int64_t H = q_proj.size(1);
  int64_t nnz = kvbuf.size(0);
  int64_t L = kvbuf.size(1);
  int64_t MAXLEN = kvbuf.size(3);
  kernel(k_proj.data_ptr(), o.data_ptr(), q_proj.data_ptr(), v_proj.data_ptr(),
         kvbuf.data_ptr(), kvidx.data_ptr(), past_len.data_ptr(), B, H, L,
         MAXLEN, layer_idx, nnz);
}

#define DEFINE_rotary_mha_decode(name)                                        \
  void name(torch::Tensor q_proj, torch::Tensor k_proj, torch::Tensor v_proj, \
            torch::Tensor o, torch::Tensor past_len, torch::Tensor kvbuf,     \
            torch::Tensor kvidx, int64_t layer_idx) {                         \
    rotary_mha_decode(launch_##name##_kernel, q_proj, k_proj, v_proj, o,      \
                      past_len, kvbuf, kvidx, layer_idx);                     \
  }

void dispatch_rotary_mha_decode(torch::Tensor q_proj, torch::Tensor k_proj,
                                torch::Tensor v_proj, torch::Tensor o,
                                torch::Tensor past_len, torch::Tensor kvbuf,
                                torch::Tensor kvidx, int64_t layer_idx) {
  CHECK_INPUT(q_proj);
  CHECK_INPUT(k_proj);
  CHECK_INPUT(v_proj);
  CHECK_INPUT(o);
  CHECK_INPUT(past_len);
  CHECK_INPUT(kvbuf);
  CHECK_INPUT(kvidx);

  CHECK_DIM(3, q_proj);
  CHECK_DIM(3, k_proj);
  CHECK_DIM(3, v_proj);
  CHECK_DIM(3, o);
  CHECK_DIM(1, past_len);
  CHECK_DIM(6, kvbuf);
  CHECK_DIM(1, kvidx);

  int64_t B = q_proj.size(0);
  int64_t H = q_proj.size(1);
  int64_t D = q_proj.size(2);
  CHECK_SHAPE(q_proj, k_proj);
  CHECK_SHAPE(q_proj, v_proj);
  CHECK_SHAPE(q_proj, o);
  CHECK_EQ(past_len.size(0), B);

  int64_t L = kvbuf.size(1);
  CHECK_EQ(kvbuf.size(2), 2);
  int64_t maxlen = kvbuf.size(3);
  CHECK_EQ(kvbuf.size(4), H);
  CHECK_EQ(kvbuf.size(5), D);
  CHECK_EQ(kvidx.size(0), B);

#define DISPATCH(num_heads, head_dim, num_layers, MAXLEN, dtype)                                                \
  if (H == num_heads && D == head_dim && L == num_layers &&                                                     \
      maxlen == MAXLEN) {                                                                                       \
    return rotary_mha_decode_kvconst(                                                                           \
        launch_rotary_mha_decode_kvconst_##num_heads##_##head_dim##_##num_layers##_##MAXLEN##_##dtype##_kernel, \
        q_proj, k_proj, v_proj, o, past_len, kvbuf, kvidx, layer_idx);                                          \
  }
  ARGS_rotary_mha_decode_kvconst(DISPATCH);
#undef DISPATCH

#define DISPATCH(head_dim, dtype)                                       \
  if (D == head_dim) {                                                  \
    return rotary_mha_decode(                                           \
        launch_rotary_mha_decode_##head_dim##_##dtype##_kernel, q_proj, \
        k_proj, v_proj, o, past_len, kvbuf, kvidx, layer_idx);          \
  }
  ARGS_rotary_mha_decode(DISPATCH);
#undef DISPATCH

  TORCH_CHECK(false, "No suitable kernel. B=", B, " H=", H, " D=", D, " L=", L,
              " maxlen=", maxlen);
}

ITER_rotary_mha_decode_kvconst(DEFINE_rotary_mha_decode_kvconst);
ITER_rotary_mha_decode(DEFINE_rotary_mha_decode);

//====== flashinfer ======

void batch_decode(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);

#define CASE(dim, _)                                                    \
  case dim:                                                             \
    FlashInferBatchDecodeKernel<dim, c_type>(                           \
        static_cast<c_type*>(o.data_ptr()),                             \
        static_cast<c_type*>(q.data_ptr()),                             \
        static_cast<c_type*>(kv_data.data_ptr()),                       \
        kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(), \
        last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx,    \
        num_heads, page_size, batch_size);                              \
    return true;

  bool ok = DISPATCH_TORCH_DTYPE(q.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", q.scalar_type(),
              " head_dim=", head_dim);

#undef CASE
}

void init_kv(torch::Tensor kv_data, torch::Tensor kv_indptr,
             torch::Tensor kv_indicies, torch::Tensor last_page_offset,
             torch::Tensor k, torch::Tensor v, torch::Tensor seqlen_indptr,
             int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);

#define CASE(dim, _)                                                           \
  case dim:                                                                    \
    FlashInferInitKvKernel<dim, c_type>(                                       \
        static_cast<c_type*>(kv_data.data_ptr()),                              \
        kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),        \
        last_page_offset.data_ptr<int32_t>(),                                  \
        static_cast<c_type*>(k.data_ptr()),                                    \
        static_cast<c_type*>(v.data_ptr()), seqlen_indptr.data_ptr<int32_t>(), \
        num_layers, layer_idx, num_heads, page_size, batch_size);              \
    return true;

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

void append_kv(torch::Tensor kv_data, torch::Tensor kv_indptr,
               torch::Tensor kv_indicies, torch::Tensor last_page_offset,
               torch::Tensor k, torch::Tensor v, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);

#define CASE(dim, _)                                                          \
  case dim:                                                                   \
    FlashInferAppendKvKernel<dim, c_type>(                                    \
        static_cast<c_type*>(kv_data.data_ptr()),                             \
        kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),       \
        last_page_offset.data_ptr<int32_t>(),                                 \
        static_cast<c_type*>(k.data_ptr()),                                   \
        static_cast<c_type*>(v.data_ptr()), num_layers, layer_idx, num_heads, \
        page_size, batch_size);                                               \
    return true;

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

//====== bgmv ======

template <typename T>
inline bool launch_bgmv_kernel(T* Y, const T* X, const T* W,
                               const int64_t* lora_indices,
                               uint16_t in_features, uint16_t out_features,
                               int64_t batch_size, int64_t num_layers,
                               int64_t layer_idx, float scale) {
  switch (pack_u16(in_features, out_features)) {
#define CASE_ONESIDE(_T, feat_in, feat_out)                           \
  case pack_u16(feat_in, feat_out):                                   \
    bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, batch_size, \
                                   num_layers, layer_idx, scale);     \
    break;
#define CASE(_T, narrow, wide)  \
  CASE_ONESIDE(T, narrow, wide) \
  CASE_ONESIDE(T, wide, narrow)

    FOR_BGMV_WIDE_NARROW(CASE, _)
#undef CASE
#undef CASE_ONESIDE
    default:
      return false;
  }

  return true;
}

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, float scale) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  int64_t num_layers = w.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  bool ok = false;
  if (h_in < 65536 && h_out < 65536) {
    switch (x.scalar_type()) {
      case at::ScalarType::Half:
        ok = launch_bgmv_kernel(static_cast<nv_half*>(y.data_ptr()),
                                static_cast<nv_half*>(x.data_ptr()),
                                static_cast<nv_half*>(w.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                num_layers, layer_idx, scale);
        break;
      case at::ScalarType::BFloat16:
        ok = launch_bgmv_kernel(static_cast<nv_bfloat16*>(y.data_ptr()),
                                static_cast<nv_bfloat16*>(x.data_ptr()),
                                static_cast<nv_bfloat16*>(w.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                num_layers, layer_idx, scale);
        break;
      default:
        break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type());
}

//====== rms_norm ======

void dispatch_rms_norm(torch::Tensor output, torch::Tensor input,
                       torch::Tensor weight, float epsilon) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);

  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_SHAPE(output, input);
  CHECK_EQ(input.size(input.dim() - 1), weight.size(0));
  CHECK_EQ(input.scalar_type(), weight.scalar_type());
  CHECK_EQ(input.scalar_type(), output.scalar_type());

  int rows = input.size(0);
  int columns = input.size(1);

  bool ok = DISPATCH_TORCH_DTYPE(input.scalar_type(), [&] {
    return rms_norm<c_type>(static_cast<c_type*>(output.data_ptr()),
                            static_cast<c_type*>(input.data_ptr()),
                            static_cast<c_type*>(weight.data_ptr()), rows,
                            columns, epsilon);
  });

  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", input.scalar_type(),
              " columns=", columns);
}

}  // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ITER_rotary_mha_decode_kvconst(DEFINE_pybind);
  ITER_rotary_mha_decode(DEFINE_pybind);
  m.def("dispatch_rotary_mha_decode", &dispatch_rotary_mha_decode,
        "dispatch_rotary_mha_decode");

  m.def("batch_decode", &batch_decode, "");
  m.def("init_kv", &init_kv, "");
  m.def("append_kv", &append_kv, "");

  m.def("dispatch_bgmv", &dispatch_bgmv, "dispatch_bgmv");

  m.def("rms_norm", &dispatch_rms_norm, "");
}
