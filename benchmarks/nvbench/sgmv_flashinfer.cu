#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "nvbench/nvbench.cuh"
#include "sgmv_flashinfer/sgmv_flashinfer.cuh"

template <typename T>
void sgmv_cpu_reference(T* y, T* x, T** w, int* s, int num_problems, int d_in, int d_out,
                        int layer_idx) {
  for (int p = 0; p < num_problems; p++) {
    for (int i = s[p]; i < s[p + 1]; i++) {
      for (int j = 0; j < d_out; j++) {
        float accum = y[i * d_out + j];
        for (int k = 0; k < d_in; k++) {
          accum += float(x[i * d_in + k]) * float(w[p][layer_idx * d_in * d_out + k * d_out + j]);
        }
        y[i * d_out + j] = accum;
      }
    }
  }
}

template <typename T>
std::vector<T> transpose(const std::vector<T>& x, uint32_t num_layers, uint32_t M, uint32_t N) {
  std::vector<T> y(x.size());
  assert(x.size() == num_layers * M * N);
  for (uint32_t l = 0; l < num_layers; l++) {
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        y[l * M * N + j * M + i] = x[l * M * N + i * N + j];
      }
    }
  }
  return std::move(y);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  float a_f32 = static_cast<float>(a);
  float b_f32 = static_cast<float>(b);
  return fabs(a_f32 - b_f32) <= (atol + rtol * fabs(b_f32));
}

uint32_t pad_to_multiple_of_16(uint32_t x) { return (x + 15) & ~15; }

void bench_sgmv(nvbench::state& state) {
  auto problem_size_s = state.get_string("problem_size");
  int num_problems = state.get_int64("num_problems");
  int d_in = state.get_int64("d_in");
  int d_out = state.get_int64("d_out");
  int num_loras = 100;
  int num_layers = 3;
  std::mt19937 gen(0xabcdabcd987);
  std::normal_distribution<float> dis;

  int problem_size;
  if (problem_size_s == "num_problems") {
    problem_size = num_problems;
  } else {
    try {
      problem_size = std::stoi(problem_size_s);
    } catch (...) {
      state.skip("problem_size is not valid");
      return;
    }
  }

  std::vector<int> s(num_problems + 1);
  std::vector<std::vector<half>> w_all(num_loras);
  s[0] = 0;
  for (size_t b = 1; b < num_problems + 1; b++) {
    s[b] = s[b - 1] + problem_size;
  }
  int batch_size = s.back();
  state.add_summary("batch_size").set_int64("value", batch_size);

  std::vector<half> x(batch_size * d_in);
  std::vector<half> y_init(batch_size * d_out);

  // random init x, w, y with normal distribution
  for (size_t i = 0; i < batch_size * d_in; i++) {
    x[i] = dis(gen);
  }
  for (size_t i = 0; i < num_loras; i++) {
    w_all[i].resize(num_layers * d_in * d_out);
    for (size_t j = 0; j < w_all[i].size(); j++) {
      w_all[i][j] = dis(gen);
    }
  }
  for (size_t i = 0; i < batch_size * d_out; i++) {
    y_init[i] = dis(gen);
  }

  // copy std vector x, w, y to thrust device vector
  thrust::device_vector<half> x_d(x.begin(), x.end());
  std::vector<thrust::device_vector<half>> w_all_d;
  for (size_t i = 0; i < num_loras; i++) {
    std::vector<half> w_all_i_trans = std::move(transpose(w_all[i], num_layers, d_in, d_out));
    w_all_d.emplace_back(w_all_i_trans.begin(), w_all_i_trans.end());
  }
  thrust::device_vector<int32_t> s_d(s.begin(), s.end());

  // build w ptr
  std::vector<half*> w;
  for (int i = 0; i < num_problems; ++i) {
    w.push_back(w_all[i].data());
  }
  std::vector<half*> w_gpu_ptr;
  for (int i = 0; i < num_problems; ++i) {
    w_gpu_ptr.push_back(thrust::raw_pointer_cast(w_all_d[i].data()));
  }
  thrust::device_vector<half*> w_d(w_gpu_ptr.begin(), w_gpu_ptr.end());

  // tmp_ptr
  thrust::device_vector<float> tmp_d(2 * 1024 * 1024);

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t D_OUT = 16;
  dim3 nthrs(32, num_warps);
  constexpr uint32_t num_stages = 2;
  constexpr uint32_t num_k_frags_per_stage = 8;
  const uint32_t num_blocks_n = d_out / 16;
  uint32_t smem =
      num_stages * sizeof(half) * num_k_frags_per_stage * 16 * 16 * (num_warps + num_blocks_n);
  cudaStream_t stream = nullptr;
  auto cooperative_kernel = flashinfer::sgmv::sgmv_shrink<true, half, int, num_warps, D_OUT>;
  auto kernel = flashinfer::sgmv::sgmv_shrink<false, half, int, num_warps, D_OUT>;

  uint32_t dev_id = 0;
  int num_blocks_per_sm = 0;
  int num_sm = 0;
  bool use_cooperative = true;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, cooperative_kernel,
                                                num_warps * 32, smem);

  const uint32_t max_grid_size = num_sm * num_blocks_per_sm;

  uint32_t chunk_size = 256;
  uint32_t num_chunks = (d_in + chunk_size - 1) / chunk_size;
  if (num_chunks * num_problems > max_grid_size) {
    use_cooperative = false;
    chunk_size = d_in;
    num_chunks = 1;
  }

  dim3 nblks(num_chunks, num_problems);

  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    // call cpu_reference function
    std::vector<half> y_cpu_ref(y_init.begin(), y_init.end());
    sgmv_cpu_reference(y_cpu_ref.data(), x.data(), w.data(), s.data(), num_problems, d_in, d_out,
                       layer_idx);

    // call sgmv function
    thrust::device_vector<half> y_d(y_init.begin(), y_init.end());

    half* y_ptr = thrust::raw_pointer_cast(y_d.data());
    half* x_ptr = thrust::raw_pointer_cast(x_d.data());
    half** w_ptr = thrust::raw_pointer_cast(w_d.data());
    int* s_ptr = thrust::raw_pointer_cast(s_d.data());
    float* tmp_ptr = thrust::raw_pointer_cast(tmp_d.data());

    void* args[] = {(void*)&y_ptr, (void*)&x_ptr,     (void*)&w_ptr,
                    (void*)&s_ptr, (void*)&tmp_ptr,   (void*)&num_problems,
                    (void*)&d_in,  (void*)&layer_idx, (void*)&chunk_size};

    cudaError_t status = cudaSuccess;
    if (use_cooperative) {
      status =
          cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks, nthrs, args, smem, stream);
    } else {
      status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem, stream);
    }
    if (status != cudaSuccess) {
      state.skip("sgmv_shrink kernel failed");
      return;
    }

    // copy thrust device_vector y_d to std vector y_h
    thrust::host_vector<half> y_h = y_d;

    // compare y_h and y_cpu_ref
    for (size_t i = 0; i < batch_size * d_out; i++) {
      if (!isclose(float(y_h[i]), float(y_cpu_ref[i]), 1e-3, 1e-3)) {
        state.skip("y_h and y_cpu_ref are not close");
        printf("layer_idx=%i, i=%zu, ref=%f, our=%f, diff=%f\n", layer_idx, i, float(y_cpu_ref[i]),
               float(y_h[i]), float(y_h[i]) - float(y_cpu_ref[i]));
        return;
      }
    }
  }
  state.add_global_memory_reads<char>(batch_size * d_in * sizeof(half)              // x
                                      + num_problems * d_in * d_out * sizeof(half)  // w
                                      + (num_problems + 1) * sizeof(int32_t)        // s
  );
  state.add_global_memory_writes<char>(batch_size * d_out * sizeof(half));

  thrust::device_vector<half> y_d(y_init.begin(), y_init.end());
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    half* y_ptr = thrust::raw_pointer_cast(y_d.data());
    half* x_ptr = thrust::raw_pointer_cast(x_d.data());
    half** w_ptr = thrust::raw_pointer_cast(w_d.data());
    int* s_ptr = thrust::raw_pointer_cast(s_d.data());
    float* tmp_ptr = thrust::raw_pointer_cast(tmp_d.data());
    int layer_idx = 0;
    void* args[] = {(void*)&y_ptr, (void*)&x_ptr,     (void*)&w_ptr,
                    (void*)&s_ptr, (void*)&tmp_ptr,   (void*)&num_problems,
                    (void*)&d_in,  (void*)&layer_idx, (void*)&chunk_size};
    cudaError_t status = cudaSuccess;
    if (use_cooperative) {
      status =
          cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks, nthrs, args, smem, stream);
    } else {
      status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem, stream);
    }
    if (status != cudaSuccess) {
      state.skip("sgmv_shrink kernel failed");
      return;
    }
  });
}

int wide_dim = 4096;
int narrow_dim = 16;

std::vector<long> num_problems = {1,  2,  3,  4,  5,  6,  7,  8,  10, 12,
                                  14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
std::vector<std::string> problem_size = {"1",  "2",  "3",  "4",  "5",  "6",  "7",
                                         "8",  "10", "12", "14", "16", "20", "24",
                                         "28", "32", "40", "48", "56", "64"};

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_shrink_NxN")
    .add_int64_axis("d_in", {wide_dim})
    .add_int64_axis("d_out", {narrow_dim})
    .add_string_axis("problem_size", {"num_problems"})
    .add_int64_axis("num_problems", num_problems);

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_shrink_bgmv")
    .add_int64_axis("d_in", {wide_dim})
    .add_int64_axis("d_out", {narrow_dim})
    .add_string_axis("problem_size", {"1"})
    .add_int64_axis("num_problems", num_problems);

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_shrink_bmm")
    .add_int64_axis("d_in", {wide_dim})
    .add_int64_axis("d_out", {narrow_dim})
    .add_string_axis("problem_size", problem_size)
    .add_int64_axis("num_problems", {1});

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_shrink_fixed-num-problems")
    .add_int64_axis("d_in", {wide_dim})
    .add_int64_axis("d_out", {narrow_dim})
    .add_string_axis("problem_size", problem_size)
    .add_int64_axis("num_problems", {8});
