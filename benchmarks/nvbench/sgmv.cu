#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <random>
#include <string>

#include "cutlass/util/device_memory.h"
#include "nvbench/nvbench.cuh"
#include "sgmv/sgmv.cuh"

template <typename T>
void sgmv_cpu_reference(T *y, T *x, T **w, int *s, int num_problems, int d_in,
                        int d_out) {
  for (int p = 0; p < num_problems; p++) {
    for (int i = s[p]; i < s[p + 1]; i++) {
      for (int j = 0; j < d_out; j++) {
        float accum = y[i * d_out + j];
        for (int k = 0; k < d_in; k++) {
          accum += float(x[i * d_in + k]) * float(w[p][k * d_out + j]);
        }
        y[i * d_out + j] = accum;
      }
    }
  }
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  float a_f32 = static_cast<float>(a);
  float b_f32 = static_cast<float>(b);
  return fabs(a_f32 - b_f32) <= (atol + rtol * fabs(b_f32));
}

void bench_sgmv(nvbench::state &state) {
  using cutlass_t = typename cutlass_dtype<half>::type;
  auto problem_size_s = state.get_string("problem_size");
  int num_problems = state.get_int64("num_problems");
  int d_in = state.get_int64("d_in");
  int d_out = state.get_int64("d_out");
  int num_loras = 100;
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
    w_all[i].resize(d_in * d_out);
    for (size_t j = 0; j < d_in * d_out; j++) {
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
    w_all_d.emplace_back(w_all[i].begin(), w_all[i].end());
  }
  thrust::device_vector<half> y_d(y_init.begin(), y_init.end());
  thrust::device_vector<int32_t> s_d(s.begin(), s.end());
  std::vector<half> y_cpu_ref(y_init.begin(), y_init.end());

  // build w ptr
  std::vector<half *> w;
  for (int i = 0; i < num_problems; ++i) {
    w.push_back(w_all[i].data());
  }
  std::vector<cutlass_t *> w_gpu_ptr;
  for (int i = 0; i < num_problems; ++i) {
    w_gpu_ptr.push_back(
        (cutlass_t *)thrust::raw_pointer_cast(w_all_d[i].data()));
  }
  thrust::device_vector<cutlass_t *> w_d(w_gpu_ptr.begin(), w_gpu_ptr.end());
  cutlass::DeviceAllocation<char> tmp_d(num_problems * 52);

  // call cpu_reference function
  sgmv_cpu_reference(y_cpu_ref.data(), x.data(), w.data(), s.data(),
                     num_problems, d_in, d_out);

  // call sgmv function
  sgmv(thrust::raw_pointer_cast(y_d.data()),
       thrust::raw_pointer_cast(x_d.data()),
       (half **)thrust::raw_pointer_cast(w_d.data()),
       thrust::raw_pointer_cast(s_d.data()), tmp_d.get(), num_problems, d_in,
       d_out);

  // copy thrust device_vector y_d to std vector y_h
  thrust::host_vector<half> y_h = y_d;

  // compare y_h and y_cpu_ref
  for (size_t i = 0; i < batch_size * d_out; i++) {
    if (!isclose(float(y_h[i]), float(y_cpu_ref[i]), 1e-3, 1e-3)) {
      state.skip("y_h and y_cpu_ref are not close");
      return;
    }
  }

  // nvbench sgmv kernel
  state.add_global_memory_reads<char>(
      batch_size * d_in * sizeof(half)              // x
      + num_problems * d_in * d_out * sizeof(half)  // w
      + (num_problems + 1) * sizeof(int32_t)        // s
  );
  state.add_global_memory_writes<char>(batch_size * d_out * sizeof(half));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    sgmv(thrust::raw_pointer_cast(y_d.data()),
         thrust::raw_pointer_cast(x_d.data()),
         (half **)thrust::raw_pointer_cast(w_d.data()),
         thrust::raw_pointer_cast(s_d.data()), tmp_d.get(), num_problems, d_in,
         d_out);
  });
}

int wide_dim = 4096;
int narrow_dim = 16;

std::vector<long> num_problems = {1,  2,  3,  4,  5,  6,  7,  8,  10, 12,
                                  14, 16, 20, 24, 28, 32, 40, 48, 56, 64};
std::vector<std::string> problem_size = {
    "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "10", "12",
    "14", "16", "20", "24", "28", "32", "40", "48", "56", "64"};

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_expand_NxN")
    .add_int64_axis("d_in", {narrow_dim})
    .add_int64_axis("d_out", {wide_dim})
    .add_string_axis("problem_size", {"num_problems"})
    .add_int64_axis("num_problems", num_problems);

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_expand_bgmv")
    .add_int64_axis("d_in", {narrow_dim})
    .add_int64_axis("d_out", {wide_dim})
    .add_string_axis("problem_size", {"1"})
    .add_int64_axis("num_problems", num_problems);

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_expand_bmm")
    .add_int64_axis("d_in", {narrow_dim})
    .add_int64_axis("d_out", {wide_dim})
    .add_string_axis("problem_size", problem_size)
    .add_int64_axis("num_problems", {1});

NVBENCH_BENCH(bench_sgmv)
    .set_name("sgmv_expand_fixed-num-problems")
    .add_int64_axis("d_in", {narrow_dim})
    .add_int64_axis("d_out", {wide_dim})
    .add_string_axis("problem_size", problem_size)
    .add_int64_axis("num_problems", {8});

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
