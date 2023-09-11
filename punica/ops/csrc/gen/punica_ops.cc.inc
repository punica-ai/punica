extern "C" void launch_rotary_mha_decode_kvconst_12_64_12_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_16_64_24_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_32_64_24_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_32_80_32_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_32_128_32_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_40_128_40_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_56_128_48_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_72_128_64_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_kvconst_96_128_96_2048_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_64_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_80_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_96_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz);
extern "C" void launch_rotary_mha_decode_128_float16_kernel(void* __restrict__ K_proj, void* __restrict__ O, void* __restrict__ Q_proj, void* __restrict__ V_proj, void* __restrict__ kvbuf, void* __restrict__ kvidx, void* __restrict__ past_len, int64_t B, int64_t H, int64_t L, int64_t MAXLEN, int64_t layer_idx, int64_t nnz);

#define ITER_rotary_mha_decode_kvconst(f) \
  f(rotary_mha_decode_kvconst_12_64_12_2048_float16) \
  f(rotary_mha_decode_kvconst_16_64_24_2048_float16) \
  f(rotary_mha_decode_kvconst_32_64_24_2048_float16) \
  f(rotary_mha_decode_kvconst_32_80_32_2048_float16) \
  f(rotary_mha_decode_kvconst_32_128_32_2048_float16) \
  f(rotary_mha_decode_kvconst_40_128_40_2048_float16) \
  f(rotary_mha_decode_kvconst_56_128_48_2048_float16) \
  f(rotary_mha_decode_kvconst_72_128_64_2048_float16) \
  f(rotary_mha_decode_kvconst_96_128_96_2048_float16) \

#define ARGS_rotary_mha_decode_kvconst(f) \
  f(12, 64, 12, 2048, float16) \
  f(16, 64, 24, 2048, float16) \
  f(32, 64, 24, 2048, float16) \
  f(32, 80, 32, 2048, float16) \
  f(32, 128, 32, 2048, float16) \
  f(40, 128, 40, 2048, float16) \
  f(56, 128, 48, 2048, float16) \
  f(72, 128, 64, 2048, float16) \
  f(96, 128, 96, 2048, float16) \

#define ITER_rotary_mha_decode(f) \
  f(rotary_mha_decode_64_float16) \
  f(rotary_mha_decode_80_float16) \
  f(rotary_mha_decode_96_float16) \
  f(rotary_mha_decode_128_float16) \

#define ARGS_rotary_mha_decode(f) \
  f(64, float16) \
  f(80, float16) \
  f(96, float16) \
  f(128, float16) \

// EOF
