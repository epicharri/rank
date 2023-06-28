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
  epic::Parameters parameters;
  u64 number_of_bits = 0ULL;
  u64 number_of_words = 0ULL;
  u64 number_of_words_padded = 0ULL;
  std::string filename = "";
  int bit_vector_content = epic::kind::one_zero_and_then_all_ones_bit_vector;
  u32 bits_in_superblock = 0U;
  u64 number_of_words_in_one_hyperblock_in_host = 0ULL;
  int fill_bit_vector_with_one_bits();
  void calculate_number_of_words();
  void calculate_number_of_words_padded();
  void calculate_hyperblock_size();
  int allocate_memory_for_data(epic::gpu::DeviceStream &);
  int create(epic::Parameters &, epic::gpu::DeviceStream &);
  int construct(epic::gpu::DeviceStream &);
  BitVector() = default;
  ~BitVector();
};

BitVector::~BitVector(){};

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
  DEBUG_CODE(fprintf(stderr, "In BitVector::create: number_of_words_padded = %" PRIu64 "\n", number_of_words_padded);)
  calculate_hyperblock_size();
  if (allocate_memory_for_data(device_stream))
    return 1;
  DEBUG_CODE(fprintf(stderr, "Memory allocated for the bit vector.\n");)
  if (rank_index.create(number_of_bits, number_of_words_padded, parameters.bits_in_superblock, parameters.rank_structure_version, device_stream))
    return 1;
  DEBUG_CODE(fprintf(stderr, "In BitVector, after rank_index.create()\n");)
  return 0;
}

int BitVector::construct(epic::gpu::DeviceStream &device_stream)
{
  fill_bit_vector_with_one_bits();
  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after fill_bit_vector_with_one_bits()\n");)

  cudaError_t err = cudaMemcpyAsync(device_data.data, host_data.data, host_data.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream);

  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after cudaMemcpy, err nro %d\n", err);)

  rank_index.construct(host_data, device_stream);

  DEBUG_CODE(fprintf(stderr, "In BitVector::construct(), after rank_index.construct()\n");) // Here???

  return 0;
}

int BitVector::fill_bit_vector_with_one_bits()
{

  for (u64 i = number_of_words - 1ULL; i < number_of_words_padded; i += 1ULL)
  {
    host_data.data[i] = 0ULL;
  }
  for (u64 i = 0ULL; i < number_of_words - 1ULL; i += 1ULL)
  {
    host_data.data[i] = 0xffff'ffff'ffff'ffffULL;
  }
  u64 last_word = 0xffff'ffff'ffff'ffffULL;
  u32 number_of_msb_in_last_word = (u32)(number_of_bits & 63ULL);
  if (number_of_msb_in_last_word == 0U)
  {
    last_word = 0ULL;
  }
  else
  {
    last_word = last_word << (64U - number_of_msb_in_last_word);
  }
  host_data.data[number_of_words - 1ULL] = last_word;
  return 0;
}

int BitVector::allocate_memory_for_data(epic::gpu::DeviceStream &device_stream)
{
  if (host_data.create(number_of_words_padded * 8ULL, epic::kind::not_write_only))
    return 1;
  if (device_data.create(number_of_words_padded * 8ULL, device_stream))
    return 1;
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
  rank(n), where n is the number of bits. If we do not add 1 and n is a multiple of 4096, the number_of_words_padded remains the same as without padding. Then, during search, rank(n) is called, and the basic block starting from index n is fetched from memory from an illegal memory address. But, if n is a multiple of 4096, n+1 is not, and the round up function sets number_of_words_padded to n+4096, which is enough to rank(n).
  */
  number_of_words_padded = (epic::utils::round_up_first_to_multiple_of_second(number_of_bits + 1ULL, 4096ULL) / 64ULL);
}

/*
int BitVector::create_for_benchmarking(u64 bits_in_bit_vector, int bit_vector_content)
{
  filename = t_filename;
  std::ifstream file(filename, std::ios::in | std::ios::ate | std::ios::binary);
  if (!file.is_open())
  {
    fprintf(stderr, "The file %s can not be opened.\n", filename);
    return 1;
  }
  u64 size_of_file_in_bytes = file.tellg();
  if (size_of_file_in_bytes < 8)
  {
    fprintf(stderr, "The size of the file %s is too small to contain any data.\n", filename);
    return 1;
  }
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer_for_size;
  buffer_for_size.reserve(8);
  file.read(&buffer_for_size[0], 8);
  u64 number_of_bits_in_file = 0ULL;
  for (int i = 0; i < 8; i++)
  {
    number_of_bits_in_file = number_of_bits_in_file | ((((uint64_t)buffer_for_size[i]) & 0xffULL) << (i * 8));
  }
  DEBUG_CODE(fprintf(stderr, "Number of bits in the file is %" PRIu64 ".\n", number_of_bits_in_file);)
  if (number_of_bits_in_file == 0ULL)
  {
    fprintf(stderr, "The first 8 bytes in the file %s indicates that there is a bit vector of size 0 in the file.\n", filename.c_str());
    return 1;
  }

  number_of_bytes = (number_of_bits_in_file + 7ULL) / 8ULL;
  if (size_of_file_in_bytes - 8ULL < number_of_bytes)
  {
    fprintf(stderr, "The first 8 bytes in the file %s indicates that the size of the bit vector is %" PRIu64 " bytes. However, there is only %" PRIu64 " bytes of data in the file.\n", filename, number_of_bytes, (size_of_file_in_bytes - 8ULL));
    return 1;
  }

  number_of_words = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 64ULL)) / 64ULL;
  number_of_words_padded = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 4096ULL) / 64ULL);
  number_of_bits = number_of_bits_in_file;
  file.close();

  return 0;
}

int BitVector::create_from_file(std::string t_filename)
{
  filename = t_filename;
  std::ifstream file(filename, std::ios::in | std::ios::ate | std::ios::binary);
  if (!file.is_open())
  {
    fprintf(stderr, "The file %s can not be opened.\n", filename);
    return 1;
  }
  u64 size_of_file_in_bytes = file.tellg();
  if (size_of_file_in_bytes < 8)
  {
    fprintf(stderr, "The size of the file %s is too small to contain any data.\n", filename);
    return 1;
  }
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer_for_size;
  buffer_for_size.reserve(8);
  file.read(&buffer_for_size[0], 8);
  u64 number_of_bits_in_file = 0ULL;
  for (int i = 0; i < 8; i++)
  {
    number_of_bits_in_file = number_of_bits_in_file | ((((uint64_t)buffer_for_size[i]) & 0xffULL) << (i * 8));
  }
  DEBUG_CODE(fprintf(stderr, "Number of bits in the file is %" PRIu64 ".\n", number_of_bits_in_file);)
  if (number_of_bits_in_file == 0ULL)
  {
    fprintf(stderr, "The first 8 bytes in the file %s indicates that there is a bit vector of size 0 in the file.\n", filename.c_str());
    return 1;
  }

  number_of_bytes = (number_of_bits_in_file + 7ULL) / 8ULL;
  if (size_of_file_in_bytes - 8ULL < number_of_bytes)
  {
    fprintf(stderr, "The first 8 bytes in the file %s indicates that the size of the bit vector is %" PRIu64 " bytes. However, there is only %" PRIu64 " bytes of data in the file.\n", filename, number_of_bytes, (size_of_file_in_bytes - 8ULL));
    return 1;
  }

  number_of_words = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 64ULL)) / 64ULL;
  number_of_words_padded = (round_up_first_to_multiple_of_second(number_of_bits_in_file, 4096ULL) / 64ULL);
  number_of_bits = number_of_bits_in_file;
  file.close();

  return 0;
}

inline int BitVector::read_bit_vector_from_file()
{
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open())
  {
    fprintf(stderr, "The file %s can not be opened.\n", filename);
    return 1;
  }
  for (u64 i = number_of_words; i < number_of_words_padded; i += 1ULL)
  {
    data[i] = 0ULL;
  }
  file.seekg(8, std::ios::beg);
  file.read(reinterpret_cast<char *>(data), number_of_bytes);
  // We should check that read is good.
  DEBUG_CODE(fprintf(stderr, "File %s read successfully.\n", filename.c_str());)
  return 0;
}

int BitVector::read()
{
  if (allocate_memory_for_data_array())
    return 1;
  if (read_bit_vector_from_file())
    return 1;
  //   Converting endianess of bit vector() is done during construction of rank data structures.
  return 0;
}

bool BitVector::system_is_little_endian()
{
  u64 number = 0x0807060504030201ULL;

  bool answer = true;
  u8 *bytes = reinterpret_cast<u8 *>(&number);
  for (u32 i = 0U; i < 8U; i++)
  {
    if (bytes[i] != (i + 1U))
      answer = false;
  }
  return answer;
}

inline u64 BitVector::swap_bytes(u64 x)
{
  return (x << 56) | ((x & 0x00'00'00'00'00'00'ff'00ULL) << 40) | ((x & 0x00'00'00'00'00'ff'00'00ULL) << 24) | ((x & 0x00'00'00'00'ff'00'00'00ULL) << 8) | ((x & 0x00'00'00'ff'00'00'00'00ULL) >> 8) | ((x & 0x00'00'ff'00'00'00'00'00ULL) >> 24) | ((x & 0x00'ff'00'00'00'00'00'00ULL) >> 40) | (x >> 56);
}

int BitVector::convert_endianess_of_bit_vector()
{
  if (system_is_little_endian())
  {
    for (u64 i = 0; i < number_of_words_padded; i += 1ULL)
    {
      data[i] = swap_bytes(data[i]);
    }
  }
  return 0;
}
*/