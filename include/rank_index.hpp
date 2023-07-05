#pragma once
#include "../include/device_array.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/host_array.hpp"
#include "../include/parameters.hpp"

struct RankIndex
{
  HostArray host_layer_0;
  HostArray host_layer_12;
  DeviceArray device_layer_0;
  DeviceArray device_layer_12;
  int rank_version = epic::kind::poppy;
  u32 bits_in_superblock = 0U;
  u32 bits_in_basicblock = 0U;
  u32 log_2_of_hyperblock_size = 32U;
  u64 number_of_words_padded_layer_0 = 0ULL;
  u64 number_of_words_padded_layer_12 = 0ULL;
  u64 number_of_words_padded_bit_vector = 0ULL;
  u64 number_of_words_in_one_hyperblock_layer_12_in_host = 0ULL;
  u64 number_of_words_in_one_hyperblock_in_host = 0ULL;
  u64 absolute_count = 0ULL;
  int create(u64, u64, u32, int, epic::Parameters &, epic::gpu::DeviceStream &);
  int init(u64, u64, u32, int);
  int allocate_memory(epic::Parameters &, epic::gpu::DeviceStream &);
  template <u32 words_in_basicblock>
  int precount_the_structures_based_on_words_in_basicblock(HostArray &, epic::gpu::DeviceStream &, u64);
  template <u32 words_in_basicblock>
  u64 popcount_basicblock(u64 *, u64);

  int construct(epic::Parameters &, HostArray &, epic::gpu::DeviceStream &);
  int compute_index(HostArray &, epic::gpu::DeviceStream &, u64);

  RankIndex() = default;
  ~RankIndex();
};

RankIndex::~RankIndex(){};

int RankIndex::create(u64 the_number_of_bits, u64 the_number_of_words_padded_bit_vector, u32 the_bits_in_superblock, int the_rank_version, epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream)
{
  if (init(the_number_of_bits, the_number_of_words_padded_bit_vector, the_bits_in_superblock, the_rank_version))
    return 1;
  if (allocate_memory(parameters, device_stream))
    return 1;
  return 0;
}

inline int RankIndex::init(u64 the_number_of_bits, u64 the_number_of_words_padded_bit_vector, u32 the_bits_in_superblock, int the_rank_version)
{
  rank_version = the_rank_version;
  bits_in_superblock = the_bits_in_superblock;
  bits_in_basicblock = the_bits_in_superblock / 4U;
  number_of_words_padded_bit_vector = the_number_of_words_padded_bit_vector;
  log_2_of_hyperblock_size = (the_bits_in_superblock == 4096) ? 31U : 32U;

  // number_of_words_in_one_hyperblock_layer_12_in_host = ((1ULL << log_2_of_hyperblock_size) / the_bits_in_superblock) / 64ULL;

  number_of_words_padded_layer_0 = 1ULL + (the_number_of_bits >> log_2_of_hyperblock_size);

  number_of_words_padded_layer_12 = 1ULL + (epic::utils::round_up_first_to_multiple_of_second(the_number_of_bits, (u64)bits_in_superblock)) / (u64)bits_in_superblock;
  return 0;
}

inline int RankIndex::allocate_memory(epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream)
{
  DEBUG_CODE(fprintf(stderr, "Layer 0 size with padding is %" PRIu64 " bytes.\n", (number_of_words_padded_layer_0 * 8ULL)););
  DEBUG_CODE(fprintf(stderr, "Layer 12 size with padding is %" PRIu64 " bytes.\n", (number_of_words_padded_layer_12 * 8ULL)););

  device_stream.start_timer();
  if (device_layer_0.create(
          number_of_words_padded_layer_0 * 8ULL, device_stream))
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_allocate_device_memory_for_L0 = device_stream.duration_in_millis();
  device_stream.start_timer();
  if (device_layer_12.create(
          number_of_words_padded_layer_12 * 8ULL, device_stream))
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_allocate_device_memory_for_L12 = device_stream.duration_in_millis();
  auto start = START_TIME;
  if (host_layer_0.create(number_of_words_padded_layer_0 * 8ULL, epic::kind::not_write_only))
    return 1;
  auto stop = STOP_TIME;
  parameters.benchmark_info.millis_allocate_host_memory_for_L0 = DURATION_IN_MILLISECONDS(start, stop);
  start = START_TIME if (host_layer_12.create(number_of_words_padded_layer_12 * 8ULL, epic::kind::not_write_only)) return 1;
  stop = STOP_TIME;
  parameters.benchmark_info.millis_allocate_host_memory_for_L12 = DURATION_IN_MILLISECONDS(start, stop);
  return 0;
}

int RankIndex::construct(epic::Parameters &parameters, HostArray &bit_vector_data, epic::gpu::DeviceStream &device_stream)
{
  DEBUG_CODE(fprintf(stderr, "In RankIndex.construct(), before compute_index()\n");)
  if (compute_index(bit_vector_data, device_stream, 0ULL))
    return 1;
  DEBUG_CODE(fprintf(stderr, "In RankIndex.construct(), after compute_index()\n");)
  DEBUG_CODE(fprintf(stderr, "In RankIndex.construct(), before cudaMemcpy L0()\n");)

  device_stream.start_timer();
  if (cudaMemcpyAsync(device_layer_0.data, host_layer_0.data, host_layer_0.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_transfer_L0_H_to_D = device_stream.duration_in_millis();

  DEBUG_CODE(fprintf(stderr, "In RankIndex.construct(), before cudaMemcpy L0()\n");)
  device_stream.start_timer();
  if (cudaMemcpyAsync(device_layer_12.data, host_layer_12.data, host_layer_12.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_transfer_L12_H_to_D = device_stream.duration_in_millis();
  return 0;
}

int RankIndex::compute_index(HostArray &bit_vector_data, epic::gpu::DeviceStream &device_stream, u64 abs_count_before = 0ULL)
{
  if (bits_in_superblock == 256)
    return precount_the_structures_based_on_words_in_basicblock<1>(bit_vector_data, device_stream, abs_count_before);
  else if (bits_in_superblock == 512)
    return precount_the_structures_based_on_words_in_basicblock<2>(bit_vector_data, device_stream, abs_count_before);
  else if (bits_in_superblock == 1024)
    return precount_the_structures_based_on_words_in_basicblock<4>(bit_vector_data, device_stream, abs_count_before);
  else if (bits_in_superblock == 2048)
    return precount_the_structures_based_on_words_in_basicblock<8>(bit_vector_data, device_stream, abs_count_before);
  else if (bits_in_superblock == 4096)
    return precount_the_structures_based_on_words_in_basicblock<16>(bit_vector_data, device_stream, abs_count_before);
  else
    return 1;
}

template <u32 words_in_basicblock>
inline u64 RankIndex::popcount_basicblock(u64 *data, u64 i)
{
  u64 popcount_of_basicblock = 0;
  u64 word;
#pragma unroll
  for (u32 j = 0U; j < words_in_basicblock; j += 1U)
  {
    word = data[i + j];
    popcount_of_basicblock += __builtin_popcountll(word);
  }
  return popcount_of_basicblock;
}

template <u32 words_in_basicblock>
int RankIndex::precount_the_structures_based_on_words_in_basicblock(HostArray &bit_vector_data, epic::gpu::DeviceStream &device_stream, u64 abs_count_before = 0ULL)
{
  u64 absolute_number_of_ones;
  u64 bits_in_hyperblock = 1ULL << log_2_of_hyperblock_size;
  u64 words_in_hyperblock = (bits_in_hyperblock / ((u64)bits_in_superblock));
  u32 one_if_sb_4096 = (u32)(bits_in_superblock == 4096U);
  u64 i_data = 0ULL;
  u64 i_layer_12 = 0ULL;
  absolute_number_of_ones = abs_count_before;
  u32 rel_count = 0U;
  for (u32 i_layer_0 = 0U; i_layer_0 < number_of_words_padded_layer_0; i_layer_0++)
  {
    host_layer_0.data[i_layer_0] = absolute_number_of_ones;
    rel_count = 0U;
    for (u32 i_layer_12_rel = 0U; i_layer_12_rel < words_in_hyperblock; i_layer_12_rel += 1U)
    {
      if (i_layer_12 >= number_of_words_padded_layer_12 - 1ULL)
      {
        host_layer_12.data[i_layer_12] = ((u64)rel_count) << (32 + one_if_sb_4096);
        return 0;
      }
      u64 countBB[4];
#pragma unroll
      for (u32 b = 0U; b < 4U; b++)
      {
        countBB[b] = popcount_basicblock<words_in_basicblock>(bit_vector_data.data, i_data + (b * words_in_basicblock));
      }
      i_data += (words_in_basicblock << 2);
      if (rank_version == epic::kind::poppy)
      {
        host_layer_12.data[i_layer_12] = (((u64)rel_count) << (32 + one_if_sb_4096)) | (((countBB[0]) << (20 + 2 * one_if_sb_4096)) | ((countBB[1]) << (10 + one_if_sb_4096)) | (countBB[2]));
      }
      else if (rank_version == epic::kind::cum_poppy)
      { // Only if superblock size is 256, 512, 1024, or 2048
        if (bits_in_superblock < 2048U)
        {
          host_layer_12.data[i_layer_12] = (((u64)rel_count) << 32) |
                                           ((((u64)countBB[0]) << 20) |
                                            (((u64)(countBB[0] + countBB[1])) << 10) |
                                            ((u64)(countBB[0] + countBB[1] + countBB[2])));
        }
        if (bits_in_superblock == 2048U)
        {
          host_layer_12.data[i_layer_12] = (((u64)rel_count) << 32) |
                                           ((((u64)countBB[0]) << 22) |
                                            (((u64)(countBB[0] + countBB[1])) << 11) |
                                            ((u64)(countBB[0] + countBB[1] + countBB[2])));
        }
      }
      i_layer_12 += 1ULL;
      u32 countSB = countBB[0] + countBB[1] + countBB[2] + countBB[3];
      absolute_number_of_ones += (u64)countSB;
      rel_count += countSB;
    }
  }
  return 0;
}
