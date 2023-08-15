#include <torch/extension.h>

#include "gen/punica_ops.cc.inc"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
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

//====== bggemv ======

template <typename F>
inline void bggemv(F kernel, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, torch::Tensor y, int64_t layer_idx) {
  int64_t B = x.size(0);
  int64_t num_loras = w.size(0);
  int64_t num_layers = w.size(1);
  kernel(indicies.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), B,
         layer_idx, num_layers, num_loras);
}

#define DEFINE_bggemv(name)                                           \
  void name(torch::Tensor x, torch::Tensor w, torch::Tensor indicies, \
            torch::Tensor y, int64_t layer_idx) {                     \
    bggemv(launch_##name##_kernel, x, w, indicies, y, layer_idx);     \
  }

void dispatch_bggemv(torch::Tensor x, torch::Tensor w, torch::Tensor indicies,
                     torch::Tensor y, int64_t layer_idx) {
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);
  CHECK_INPUT(y);

  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);
  CHECK_DIM(2, y);

  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  CHECK_EQ(w.size(2), h_in);
  CHECK_EQ(w.size(3), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));

#define DISPATCH(hi, ho, dtype)                                       \
  if (h_in == hi && h_out == ho) {                                    \
    return bggemv(launch_bggemv_##hi##_##ho##_##dtype##_kernel, x, w, \
                  indicies, y, layer_idx);                            \
  }
  ARGS_bggemv(DISPATCH);
#undef DISPATCH

  TORCH_CHECK(false, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out);
}

ITER_bggemv(DEFINE_bggemv);

}  // namespace

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ITER_rotary_mha_decode_kvconst(DEFINE_pybind);
  ITER_rotary_mha_decode(DEFINE_pybind);
  m.def("dispatch_rotary_mha_decode", &dispatch_rotary_mha_decode,
        "dispatch_rotary_mha_decode");

  ITER_bggemv(DEFINE_pybind);
  m.def("dispatch_bggemv", &dispatch_bggemv, "dispatch_bggemv");
}
