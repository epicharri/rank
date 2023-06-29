#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/gpu/kernel_launcher.hpp"
#include "../include/parameters.hpp"

namespace epic
{

  float call_rank_search(epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream, BitVector &bit_vector, RankIndex &rank_index, DeviceArray &positions_in_and_results_out, u64 number_of_positions, bool device_is_nvidia_a100)
  {
    if (device_is_nvidia_a100)
    {
      cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, parameters.device_set_limit_presearch); // only in Nvidia A100
    }
    u64 block_size, grid_size;
    block_size = parameters.threads_per_block;
    grid_size = (number_of_positions + block_size - 1ULL) / block_size;

    if (grid_size == 0ULL)
      grid_size = 1ULL;
    int rank_version = parameters.rank_structure_version;

    switch (parameters.bits_in_superblock)
    {
    case 256:
      if (rank_version == kind::poppy)
        return epic::gpu::launch_rank_search_kernel<256, false, kind::poppy>(
            grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      else if (rank_version == kind::cum_poppy)
        return epic::gpu::launch_rank_search_kernel<256, false, kind::cum_poppy>(
            grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      break;
    case 512:
      if (rank_version == kind::poppy)
        return epic::gpu::launch_rank_search_kernel<512, false, kind::poppy>(
            grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      else if (rank_version == kind::cum_poppy)
        return epic::gpu::launch_rank_search_kernel<512, false, kind::cum_poppy>(
            grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      break;
    case 1024:
      if (parameters.with_shuffles)
      {
        if (rank_version == kind::poppy)
          return epic::gpu::launch_rank_search_kernel<1024, true, kind::poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
        else if (rank_version == kind::cum_poppy)
          return epic::gpu::launch_rank_search_kernel<1024, true, kind::cum_poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      }
      else
      {
        if (rank_version == kind::poppy)
          return epic::gpu::launch_rank_search_kernel<1024, false, kind::poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
        else if (rank_version == kind::cum_poppy)
          return epic::gpu::launch_rank_search_kernel<1024, false, kind::cum_poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      }
      break;
    case 2048:
      if (parameters.with_shuffles)
      {
        if (rank_version == kind::poppy)
          return epic::gpu::launch_rank_search_kernel<2048, true, kind::poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
        else if (rank_version == kind::cum_poppy)
          return epic::gpu::launch_rank_search_kernel<2048, true, kind::cum_poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      }
      else
      {
        if (rank_version == kind::poppy)
          return epic::gpu::launch_rank_search_kernel<2048, false, kind::poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
        else if (rank_version == kind::cum_poppy)
          return epic::gpu::launch_rank_search_kernel<2048, false, kind::cum_poppy>(
              grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      }
      break;
    case 4096:
      if (rank_version == kind::poppy)
        return epic::gpu::launch_rank_search_kernel<4096, false, kind::poppy>(
            grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);
      // Super block size 4096 with cum_poppy is not supported
      else if (rank_version == kind::cum_poppy)
        printf("Super block size 4096 with cumulative poppy is not supported, and the search is not completed.");
      // return epic::gpu::launch_rank_search_kernel<4096, false, kind::poppy>(grid_size, block_size, device_stream, bit_vector, rank_index, positions_in_and_results_out, number_of_positions);

      break;
    }
    return 0.0;
  }

}