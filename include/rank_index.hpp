#pragma once
#include "../include/cuda.hpp"
#include "../include/device_array.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/host_array.hpp"
#include <omp.h>

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
  int create(u64, u64, u32, int);
  int init(u64, u64, u32, int);
  int allocate_memory();
  // int compute_rank_index(/*Bitvector needed somehow!*/);
  template <u32 words_in_basicblock>
  int precount_the_structures_based_on_words_in_basicblock(HostArray &, u64);
  template <u32 words_in_basicblock>
  u64 popcount_basicblock(u64 *, u64);

  int construct(HostArray &, u64);
  int compute_index(HostArray &, u64);

  RankIndex() = default;
  ~RankIndex();
};

RankIndex::~RankIndex(){};

int RankIndex::create(u64 the_number_of_bits, u64 the_number_of_words_padded_bit_vector, u32 the_bits_in_superblock, int the_rank_version)
{
  if (init(the_number_of_bits, the_number_of_words_padded_bit_vector, the_bits_in_superblock, the_rank_version))
    return 1;
  if (allocate_memory())
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

inline int RankIndex::allocate_memory()
{
  if (device_layer_0.create(
          number_of_words_padded_layer_0 * 8ULL) |
      device_layer_12.create(
          number_of_words_padded_layer_12 * 8ULL))
  {
    return 1;
  }
  if (host_layer_0.create(
          number_of_words_padded_layer_0 * 8ULL, epic::kind::not_write_only) |
      host_layer_12.create(
          number_of_words_in_one_hyperblock_layer_12_in_host * 8ULL, epic::kind::not_write_only))
  {
    return 1;
  }
  return 0;
}

int RankIndex::construct(HostArray &bit_vector_data, u64 abs_count_before = 0ULL)
{
  if (compute_index, bit_vector_data)
    return 1;
  if (CHECK(cudaMemcpy(device_layer_0, host_layer_0, host_layer_0.size_in_bytes, cudaMemcpyHostToDevice)))
    return 1;
  if (CHECK(cudaMemcpy(device_layer_0, host_layer_0, host_layer_0.size_in_bytes, cudaMemcpyHostToDevice)))
    return 1;
  return 0;
}

int RankIndex::compute_index(HostArray &bit_vector_data, u64 abs_count_before = 0ULL)
{
  if (bits_in_superblock == 256)
    return precount_the_structures_based_on_words_in_basicblock<1>(bit_vector_data, abs_count_before);
  else if (bits_in_superblock == 512)
    return precount_the_structures_based_on_words_in_basicblock<2>(bit_vector_data, abs_count_before);
  else if (bits_in_superblock == 1024)
    return precount_the_structures_based_on_words_in_basicblock<4>(bit_vector_data, abs_count_before);
  else if (bits_in_superblock == 2048)
    return precount_the_structures_based_on_words_in_basicblock<8>(bit_vector_data, abs_count_before);
  else if (bits_in_superblock == 4096)
    return precount_the_structures_based_on_words_in_basicblock<16>(bit_vector_data, abs_count_before);
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
int RankIndex::precount_the_structures_based_on_words_in_basicblock(HostArray &bit_vector_data, u64 abs_count_before = 0ULL)
{
  u64 bits_in_hyperblock = 1ULL << log_2_of_hyperblock_size;
  u64 words_in_hyperblock = (bits_in_hyperblock / ((u64)bits_in_superblock));
  u32 one_if_sb_4096 = (u32)(bits_in_superblock == 4096U);
  u64 i_data = 0ULL;
  u64 i_layer_12 = 0ULL;
  absolute_number_of_ones = abs_count_before;
  u32 rel_count = 0U;
  for (u32 i_layer_0 = 0U; i_layer_0 < number_of_words_padded_layer_0; i_layer_0++)
  {
    layer_0[i_layer_0] = absolute_number_of_ones;
    rel_count = 0U;
    for (u32 i_layer_12_rel = 0U; i_layer_12_rel < words_in_hyperblock; i_layer_12_rel += 1U)
    {
      if (i_layer_12 >= number_of_words_padded_layer_12 - 1ULL)
      {
        layer_12[i_layer_12] = ((u64)rel_count) << (32 + one_if_sb_4096);
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
        layer_12[i_layer_12] = (((u64)rel_count) << (32 + one_if_sb_4096)) | (((countBB[0]) << (20 + 2 * one_if_sb_4096)) | ((countBB[1]) << (10 + one_if_sb_4096)) | (countBB[2]));
      }
      else if (rank_version == epic::kind::cum_poppy)
      { // Only if superblock size is 256, 512, 1024, or 2048
        if (bits_in_superblock < 2048U)
        {
          layer_12[i_layer_12] = (((u64)rel_count) << 32) |
                                 ((((u64)countBB[0]) << 20) |
                                  (((u64)(countBB[0] + countBB[1])) << 10) |
                                  ((u64)(countBB[0] + countBB[1] + countBB[2])));
        }
        if (bits_in_superblock == 2048U)
        {
          layer_12[i_layer_12] = (((u64)rel_count) << 32) |
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

/*
template <u32 words_in_basicblock>
int RankIndex::compute_rank_index_based_on_words_in_basic_block(u64 abs_count_before = 0ULL)
{
  u64 bits_in_hyperblock = 1ULL << log_2_of_hyperblock_size;
  u64 words_in_hyperblock = (bits_in_hyperblock / ((u64)bits_in_superblock));
  u32 one_if_sb_4096 = (u32)(bits_in_superblock == 4096U);
  u64 i_data = 0ULL;
  u64 i_layer_12 = 0ULL;
  absolute_number_of_ones = abs_count_before;
  u32 rel_count = 0U;
  for (u32 i_layer_0 = 0U; i_layer_0 < number_of_words_padded_layer_0; i_layer_0++)
  {
    layer_0[i_layer_0] = absolute_number_of_ones;
    rel_count = 0U;
    for (u32 i_layer_12_rel = 0U; i_layer_12_rel < words_in_hyperblock; i_layer_12_rel += 1U)
    {
      if (i_layer_12 >= number_of_words_padded_layer_12 - 1ULL)
      {
        layer_12[i_layer_12] = ((u64)rel_count) << (32 + one_if_sb_4096);
        return 0;
      }
      u64 countBB[4];
#pragma unroll
      for (u32 b = 0U; b < 4U; b++)
      {
        countBB[b] = popcount_basicblock<words_in_basicblock>(bit_vector.data, i_data + (b * words_in_basicblock));
      }
      i_data += (words_in_basicblock << 2);
      if (rank_version == epic::kind::poppy)
      {
        layer_12[i_layer_12] = (((u64)rel_count) << (32 + one_if_sb_4096)) | (((countBB[0]) << (20 + 2 * one_if_sb_4096)) | ((countBB[1]) << (10 + one_if_sb_4096)) | (countBB[2]));
      }
      else if (rank_version == epic::kind::cum_poppy)
      { // Only if superblock size is 256, 512, 1024, or 2048
        if (bits_in_superblock < 2048U)
        {
          layer_12[i_layer_12] = (((u64)rel_count) << 32) |
                                 ((((u64)countBB[0]) << 20) |
                                  (((u64)(countBB[0] + countBB[1])) << 10) |
                                  ((u64)(countBB[0] + countBB[1] + countBB[2])));
        }
        if (bits_in_superblock == 2048U)
        {
          layer_12[i_layer_12] = (((u64)rel_count) << 32) |
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
*/