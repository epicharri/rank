#pragma once
#include "../include/device_array.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include "../include/host_array.hpp"
#include "../include/parameters.hpp"
#include "../include/rank_index.hpp"
#include "../include/utils/helpers.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct BitVector
{
  HostArray host_data;
  DeviceArray device_data;
  RankIndex rank_index;
  u64 number_of_bits = 0ULL;
  u64 number_of_words = 0ULL;
  u64 number_of_words_padded = 0ULL;
  std::string filename = "";
  int bit_vector_content = epic::kind::one_zero_and_then_all_ones_bit_vector;
  u32 bits_in_superblock = 0U;
  u64 number_of_words_in_one_hyperblock_in_host = 0ULL;
  int destruct_host_data();
  int fill_bit_vector_with_one_bits();
  int fill_bit_vector_with_random_bits();
  void calculate_number_of_words();
  void calculate_number_of_words_padded();
  void calculate_hyperblock_size();
  int allocate_memory_for_data(epic::Parameters &, epic::gpu::DeviceStream &);
  int create(epic::Parameters &, epic::gpu::DeviceStream &);
  int construct(epic::Parameters &, epic::gpu::DeviceStream &);
  BitVector() = default;
  ~BitVector();
};

BitVector::~BitVector(){};

int BitVector::destruct_host_data()
{
  return host_data.destruct();
}

int BitVector::create(epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream)
{
  bit_vector_content = parameters.bit_vector_data_type;
  if (bit_vector_content == epic::kind::from_file_bit_vector)
  {
    printf("Currently, this program is for benchmarking only, without a support to use bit vectors from a file.\n");
    return 1;
  }
  bits_in_superblock = parameters.bits_in_superblock;
  number_of_bits = parameters.bits_in_bit_vector;
  DEBUG_CODE(fprintf(stderr, "In BitVector::create(): parameters.number_of_bits = %" PRIu64 "\n", number_of_bits);)
  calculate_number_of_words();
  calculate_number_of_words_padded();
  parameters.benchmark_info.number_of_words_in_bit_vector = number_of_words;
  parameters.benchmark_info.number_of_bytes_padded_bit_vector = number_of_words_padded * sizeof(u64);
  DEBUG_CODE(fprintf(stderr, "In BitVector::create: number_of_words_padded = %" PRIu64 "\n", number_of_words_padded);)
  calculate_hyperblock_size();
  if (allocate_memory_for_data(parameters, device_stream))
    return 1;
  DEBUG_CODE(fprintf(stderr, "Memory allocated for the bit vector.\n");)
  if (rank_index.create(number_of_bits, number_of_words_padded, parameters.bits_in_superblock, parameters.rank_structure_version, device_stream))
    return 1;
  parameters.benchmark_info.number_of_bytes_padded_layer_0 = rank_index.host_layer_0.size_in_bytes;
  parameters.benchmark_info.number_of_bytes_padded_layer_12 = rank_index.host_layer_12.size_in_bytes;
  DEBUG_CODE(fprintf(stderr, "In BitVector, after rank_index.create()\n");)
  return 0;
}

int BitVector::construct(epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream)
{
  auto start = START_TIME;
  if (bit_vector_content == epic::kind::one_zero_and_then_all_ones_bit_vector)
    fill_bit_vector_with_one_bits();
  else if (bit_vector_content == epic::kind::random_bit_vector)
    fill_bit_vector_with_random_bits();

  auto stop = STOP_TIME;
  float millis = DURATION_IN_MILLISECONDS(start, stop);
  DEBUG_CODE(fprintf(stderr, "Creating the bit vector takes %f ms\n", millis);)
  DEBUG_CODE(fprintf(stderr, "Bit vector size with padding is %" PRIu64 " bytes.\n", (number_of_words_padded * 8ULL)););
  DEBUG_CODE(fprintf(stderr, "Bit vector size with padding, in bits, is %" PRIu64 " bits.\n", (number_of_words_padded * 64ULL)););

  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after fill_bit_vector_with_one_bits()\n");)

  device_stream.start_timer();
  cudaError_t err = cudaMemcpyAsync(device_data.data, host_data.data, host_data.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream);
  device_stream.stop_timer();
  millis = device_stream.duration_in_millis();
  parameters.benchmark_info.millis_transfer_bit_vector_H_to_D = millis;
  DEBUG_CODE(fprintf(stderr, "Transferring the bit vector from host to device takes %f ms\n", millis);)
  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after cudaMemcpy, err nro %d\n", err);)

  rank_index.construct(parameters, host_data, device_stream);

  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after rank_index.construct()\n");) // Here???

  return 0;
}

// Filled with 32-bit signed positive integers. Thus, every 32nd bit is 0.
int BitVector::fill_bit_vector_with_random_bits()
{
  srandom(1);
  for (u64 i = number_of_words - 1ULL; i < number_of_words_padded; i += 1ULL)
  {
    host_data.data[i] = 0ULL;
  }
  for (u64 j = 0ULL; j < number_of_words; j += 1ULL)
  {
    host_data.data[j] = (((u64)random()) << 32) | ((u64)random());
  }
  u32 number_of_msb_in_last_word = (u32)(number_of_bits & 63ULL);
  u64 mask = 0xffff'ffff'ffff'ffff;
  if (number_of_msb_in_last_word > 0U) //  If 0, then actually the number of msb in last word is 64.
  {
    mask = mask << (64U - number_of_msb_in_last_word);
  }
  host_data.data[number_of_words - 1ULL] &= mask;

  return 0;
}

// First bit is 0, other bits are ones.
int BitVector::fill_bit_vector_with_one_bits()
{

  for (u64 i = number_of_words - 1ULL; i < number_of_words_padded; i += 1ULL)
  {
    host_data.data[i] = 0ULL;
  }
  host_data.data[0] = 0x7fff'ffff'ffff'ffffULL;
  for (u64 i = 1ULL; i < number_of_words - 1ULL; i += 1ULL)
  {
    host_data.data[i] = 0xffff'ffff'ffff'ffffULL;
  }
  u64 last_word = 0xffff'ffff'ffff'ffffULL;
  u32 number_of_msb_in_last_word = (u32)(number_of_bits & 63ULL);
  if (number_of_msb_in_last_word > 0U) // If 0, then all bits are considered.
  {
    last_word = last_word << (64U - number_of_msb_in_last_word);
  }
  host_data.data[number_of_words - 1ULL] = last_word;
  return 0;
}

int BitVector::allocate_memory_for_data(epic::Parameters &parameters, epic::gpu::DeviceStream &device_stream)
{
  auto start = START_TIME;
  if (host_data.create(number_of_words_padded * 8ULL, epic::kind::not_write_only))
    return 1;
  auto stop = STOP_TIME;
  parameters.benchmark_info.millis_allocate_host_memory_for_bit_vector = DURATION_IN_MILLISECONDS(start, stop);
  device_stream.start_timer();
  if (device_data.create(number_of_words_padded * 8ULL, device_stream))
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_allocate_device_memory_for_bit_vector = device_stream.duration_in_millis();
  return 0;
}

inline void BitVector::calculate_hyperblock_size()
{
  u64 log_2_of_hyperblock_size = (bits_in_superblock == 4096) ? 31 : 32;

  number_of_words_in_one_hyperblock_in_host = (1ULL << log_2_of_hyperblock_size) / 64ULL;
  if (number_of_words_padded < number_of_words_in_one_hyperblock_in_host)
  {
    number_of_words_in_one_hyperblock_in_host = number_of_words_padded;
  }
}

inline void BitVector::calculate_number_of_words()
{
  number_of_words = (epic::utils::round_up_first_to_multiple_of_second<u64>(number_of_bits, 64ULL)) / 64ULL;
}

inline void BitVector::calculate_number_of_words_padded()
{
  /* Number of bits is rounded up to be multiple of 4096, which is the highest supported
  superblock size in bits. This is done to get rid of branching in the end of the
  bit vector while calculating the rank index.
  The expression number_of_bits + 1ULL is needed to allow
  rank(n), where n is the number of bits. If we do not add 1 and n is a multiple of 4096, the number_of_words_padded remains the same as without padding. Then, during search, rank(n) is called, and the basic block starting from index n is fetched from memory from an illegal memory address. But, if n is a multiple of 4096, n+1 is not, and the round up function sets number_of_words_padded to n+4096, which is enough for rank(n).
  */
  number_of_words_padded = (epic::utils::round_up_first_to_multiple_of_second(number_of_bits + 1ULL, 4096ULL) / 64ULL);
}
