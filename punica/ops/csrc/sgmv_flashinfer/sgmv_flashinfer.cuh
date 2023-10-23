#include "cp_async.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace sgmv {

template <typename T, typename IdType, uint32_t num_warps, uint32_t d_in, uint32_t d_out>
__global__ void sgmv_shrink(T* y, T* x, T* w, IdType* s, IdType* I, uint32_t num_problems) {
  uint32_t problem_id = blockIdx.x;
  uint32_t s_start = s[problem_id], s_end = s[problem_id + 1];
  constexpr uint32_t num_warps = d_out / 16;
  constexpr uint32_t num_k_frags = 8;
  constexpr uint32_t block_k = num_k_frags * 16;
  constexpr uint32_t num_cells_per_block_k = block_k / cell_capacity<T>();
  constexpr uint32_t block_m = 16;
  constexpr uint32_t block_n = 16;
  constexpr uint32_t num_iterations = d_in / block_k;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  extern __shared__ uint8_t smem[];

  smem_t x_smem<num_cells_per_block_k>(smem);
  smem_t w_smem<num_cells_per_block_k>(smem + sizeof(T) * num_warps * block_m * block_k);
  smem_t o_smem<num_cells_per_block_k>(smem + 2 * sizeof(T) * num_warps * block_n * block_k);

  uint32_t a_frag[num_k_frags][4];
  uint32_t b_frag[num_k_frags][num_warps][4];
  float o_frag[8];

  for (uint32_t i = 0; i < (s_end - s_start + block_m - 1) / bx; ++i) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[reg_id] = 0.f;
    }

    uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
    T* x_ptr = x + row_idx * d_in + (4 * 0 + tx % 4) * cell_capacity<T>();
    x_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_m + tx / 4, tx % 4);
    // pre-load x_smem, w_smem
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
        x_smem.load_128b_async(x_ptr, row_idx < s_end);
        x_ptr += 4 * cell_capacity<T>();
        x_smem.offset += 8;
      }
      row_idx += 8;
      x_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_stages_per_stage;
      x_smem.offset += 8 * num_cells_per_block_k - 4 * num_k_frags;
    }
    row_idx -= 8;

    T* w_ptr = w + I[problem_id] * d_in * d_out + (ty * 16 + tx / 4) * d_in +
               (4 * 0 + tx % 4) * cell_capacity<T>();
    w_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_n + tx / 4, tx % 4);
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
        w_smem.load_128b_async(w_ptr);
        w_ptr += 4 * cell_capacity<T>();
        w_smem.offset += 8;
      }
      w_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_stages_per_stage;
      w_smem.offset += 8 * num_cells_per_block_k - 4 * num_k_frags;
    }
    cp_async::commit_group();

#pragma unroll
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<0>();
      __syncthreads();
      // load a_frag
      x_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_m + tx % 16, tx / 16);
#pragma unroll
      for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
        x_smem.ldmatrix_m8n8x4(x_frag[fk]);
        x_smem.offset = (x_smem.offset ^ 0x2) + (fk & 0x1) * 8;
      }

      // load b_frag
#prama unroll
      for (uint32_t fn = 0; fn < num_warps; ++fn) {
        w_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(fn * block_n + 8 * (tx / 16) + tx % 8, (tx % 16) / 8);
#pragma unroll
        for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
          w_smem.ldmatrix_m8n8x4(b_frag[fk][fn]);
          w_smem.offset = (w_smem.offset ^ 0x2) + (fk & 0x1) * 8;
        }
      }

      // load next stage
      if (iter + 1 < num_iterations) {
        x_ptr = x + row_idx * d_in + (2 * num_k_frags * (iter + 1) + tx % 4) * cell_capacity<T>();
        x_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_m + tx / 4, tx % 4);
        // pre-load x_smem, w_smem
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
            x_smem.load_128b_async(x_ptr, row_idx < s_end);
            x_ptr += 4 * cell_capacity<T>();
            x_smem.offset += 8;
          }
          row_idx += 8;
          x_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_stages_per_stage;
          x_smem.offset += 8 * num_cells_per_block_k - 4 * num_k_frags;
        }
        row_idx -= 8;

        w_ptr = w + I[problem_id] * d_in * d_out + (ty * 16 + tx / 4) * d_in +
                (2 * num_k_frags * (iter + 1) + tx % 4) * cell_capacity<T>();
        w_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_n + tx / 4, tx % 4);
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
            w_smem.load_128b_async(w_ptr);
            w_ptr += 4 * cell_capacity<T>();
            w_smem.offset += 8;
          }
          w_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_stages_per_stage;
          w_smem.offset += 8 * num_cells_per_block_k - 4 * num_k_frags;
        }
      }

      // compute o_frag
#pragma unroll
      for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
#pragma unroll
        for (uint32_t fn = 0; fn < num_warps; ++fn) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<T>(o_frag[fn], a_frag[fk], b_frag[fk][fn]);
        }
      }
    }
    cp_async::wait_group<0>();
    __syncthreads();

    // store o_frag
    o_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_m + tx / 4, 0);
#pragma unroll
    for (uint32_t fn = 0; fn < num_warps; ++fn) {
      vec_cast<T, float, 2>((T*)(o_smem.base + o_smem.offset) + (tx % 4) * 2, &o_frag[fn][0]);
      vec_cast<T, float, 2>((T*)(o_smem.base + (o_smem.offset ^ 0x1)) + (tx % 4) * 2, &o_frag[fn][2]);
      vec_cast<T, float, 2>((T*)(o_smem.base + o_smem.offset + 8 * num_cells_per_block_k) + (tx % 4) * 2, &o_frag[fn][4]);
      vec_cast<T, float, 2>((T*)(o_smem.base + (o_smem.offset ^ 0x1) + 8 * num_cells_per_block_k) + (tx % 4) * 2, &o_frag[fn][6]);
      o_smem.offset = (o_smem.offset ^ 0x2) + (fn & 0x1) * 8;
    }

    // store y
    row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
    o_smem.offset = smem_t::get_permuted_offset<num_cells_per_block_k>(ty * block_m + tx / 4, tx % 4);
    T* o_ptr = o + row_idx * d_out + (4 * 0 + tx % 4) * cell_capacity<T>();
 #pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fno = 0; fno < num_warps / 2; ++fno) {
        o_smem.store_128b(o_ptr, row_idx < s_end);
        o_ptr += 4 * cell_capacity<T>();
        o_smem.offset += 8;
      }
      row_idx += 8;
      o_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_stages_per_stage;
      o_smem.offset += 8 * num_cells_per_block_k - 4 * num_warps;
    }   
  }
}

}  // namespace sgmv
}  // namespace flashinfer