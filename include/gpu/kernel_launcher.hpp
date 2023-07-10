#pragma once
#include "../bit_vector.hpp"
#include "../enums.hpp"
#include "../globals.hpp"
#include "../rank_index.hpp"
#include "cuda.hpp"
#include "device_stream.hpp"
#include "rank_search_kernel.hpp"
#include <cstdlib>

namespace epic
{
  namespace gpu
  {

    template <int superblock_size,
              bool shuffles = false,
              int rank_version = epic::kind::poppy>
    inline float launch_rank_search_kernel(
        std::size_t grid_size,
        std::size_t block_size,
        epic::gpu::DeviceStream &device_stream,
        BitVector &bit_vector,
        RankIndex &rank_index,
        DeviceArray &positions_in_and_results_out,
        u64 number_of_positions)
    {
      device_stream.start_timer();
      epic::gpu::rank_search<superblock_size, shuffles, rank_version><<<dim3(grid_size), dim3(block_size), 0, device_stream.stream>>>(
          bit_vector.device_data.data,
          rank_index.device_layer_0.data,
          rank_index.device_layer_12.data,
          positions_in_and_results_out.data,
          number_of_positions);
      device_stream.stop_timer();

      return device_stream.duration_in_millis();
    }
  }
}
