#pragma once
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "../enums.hpp"
#include "rank_kernel.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

namespace epic
{
  namespace gpu
  {
    template <int superblock_size, bool shuffles = false, int rank_version = epic::kind::poppy>
    __global__ void rank_search(
        u64 *bit_vector, u64 *L0, u64 *L12, u64 *positions_in_and_results_out, u64 number_of_positions)
    {
      u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < number_of_positions)
      {
        positions_in_and_results_out[idx] = epic::gpu::rank<superblock_size, shuffles, rank_version>(bit_vector, L0, L12, positions_in_and_results_out[idx]);
      }
    }
  }
}
