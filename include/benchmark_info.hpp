#pragma once
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include <string>

struct BenchmarkInfo
{
  u64 number_of_bits_in_bit_vector = 0ULL;
  u64 number_of_words_in_bit_vector = 0ULL;
  u64 bits_in_superblock = 0ULL;
  u64 number_of_bytes_padded_bit_vector = 0ULL;
  u64 number_of_bytes_padded_layer_0 = 0ULL;
  u64 number_of_bytes_padded_layer_12 = 0ULL;

  u64 number_of_positions = 0ULL;
  u64 start_position = 0ULL;

  u32 threads_per_block = 0ULL;
  u64 number_of_errors = 0ULL;
  std::string rank_structure_version = "";
  std::string bit_vector_content = "";
  std::string positions_type = "";
  std::string shuffles = "";
  std::string store_results_info = "";

  float millis_allocate_host_memory_for_bit_vector = 0.0;
  float millis_allocate_device_memory_for_bit_vector = 0.0;
  float millis_allocate_host_memory_for_L0 = 0.0;
  float millis_allocate_device_memory_for_L0 = 0.0;
  float millis_allocate_host_memory_for_L12 = 0.0;
  float millis_allocate_device_memory_for_L12 = 0.0;
  float millis_free_host_memory_of_bit_vector = 0.0;
  float millis_transfer_bit_vector_H_to_D = 0.0;
  float millis_transfer_positions_H_to_D = 0.0;
  float millis_transfer_results_D_to_H = 0.0;
  float millis_search = 0.0;

  float get_nanos_per_query();
  u64 get_number_of_bits_padded_bit_vector();
  u64 get_number_of_bytes_padded_bit_vector();
  u64 get_number_of_words_padded_bit_vector();
  u64 get_number_of_words_in_bit_vector();

  u64 get_number_of_bits_padded_layer_0();
  u64 get_number_of_bits_padded_layer_12();

  u64 get_number_of_bytes_padded_layer_0();
  u64 get_number_of_bytes_padded_layer_12();

  u64 get_number_of_words_padded_layer_0();
  u64 get_number_of_words_padded_layer_12();

  int print_info();
  BenchmarkInfo() = default;
};

int BenchmarkInfo::print_info()
{
  fprintf(stderr, "number_of_bits_in_bit_vector \n%" PRIu64 "\nnumber_of_words_in_bit_vector \n%" PRIu64 "\nbits_in_superblock \n%" PRIu64 "\nnumber_of_bytes_padded_bit_vector \n%" PRIu64 "\nnumber_of_bytes_padded_layer_0 \n%" PRIu64 "\nnumber_of_bytes_padded_layer_12 \n%" PRIu64 "\nnumber_of_positions \n%" PRIu64 "\nstart_position \n%" PRIu64 "\nthreads_per_block \n%" PRIu32 "\nnumber_of_errors \n%" PRIu64 "\nrank_structure_version \n%s\nbit_vector_content\n%s\npositions_type\n%s\nshuffles\n%s\nstore_results_info\n%s\n",
          number_of_bits_in_bit_vector,
          number_of_words_in_bit_vector,
          bits_in_superblock,
          number_of_bytes_padded_bit_vector,
          number_of_bytes_padded_layer_0,
          number_of_bytes_padded_layer_12,
          number_of_positions,
          start_position,
          threads_per_block,
          number_of_errors,
          rank_structure_version.c_str(),
          bit_vector_content.c_str(),
          positions_type.c_str(),
          shuffles.c_str(),
          store_results_info.c_str());

  fprintf(stderr, "millis_allocate_host_memory_for_bit_vector\n%s\nmillis_allocate_device_memory_for_bit_vector\n%s\nmillis_allocate_host_memory_for_L0\n%s\nmillis_allocate_device_memory_for_L0\n%s\nmillis_allocate_host_memory_for_L12\n%s\nmillis_allocate_device_memory_for_L12\n%s\nmillis_free_host_memory_of_bit_vector\n%s\nmillis_transfer_bit_vector_H_to_D\n%s\nmillis_transfer_positions_H_to_D\n%s\nmillis_transfer_results_D_to_H\n%s\nmillis_search\n%s\nnanos_per_query\n%s\n", millis_allocate_host_memory_for_bit_vector, millis_allocate_device_memory_for_bit_vector, millis_allocate_host_memory_for_L0, millis_allocate_device_memory_for_L0, millis_allocate_host_memory_for_L12, millis_allocate_device_memory_for_L12, millis_free_host_memory_of_bit_vector, millis_transfer_bit_vector_H_to_D, millis_transfer_positions_H_to_D, millis_transfer_results_D_to_H, millis_search, get_nanos_per_query());

  fprintf(stderr, "number_of_bits_padded_bit_vector\n%" PRIu64 "\nnumber_of_bytes_padded_bit_vector\n%" PRIu64 "\nnumber_of_words_padded_bit_vector\n%" PRIu64 "\nnumber_of_words_in_bit_vector\n%" PRIu64 "\nnumber_of_bits_padded_layer_0\n%" PRIu64 "\nnumber_of_bits_padded_layer_12\n%" PRIu64 "\nnumber_of_bytes_padded_layer_0\n%" PRIu64 "\nnumber_of_bytes_padded_layer_12\n%" PRIu64 "\nnumber_of_words_padded_layer_0\n%" PRIu64 "\nnumber_of_words_padded_layer_12\n%" PRIu64 "\n", get_number_of_bits_padded_bit_vector(), get_number_of_bytes_padded_bit_vector(), get_number_of_words_padded_bit_vector(), get_number_of_words_in_bit_vector(), get_number_of_bits_padded_layer_0(), get_number_of_bits_padded_layer_12(), get_number_of_bytes_padded_layer_0(), get_number_of_bytes_padded_layer_12(), get_number_of_words_padded_layer_0(), get_number_of_words_padded_layer_12());
  return 0;
}

float BenchmarkInfo::get_nanos_per_query()
{
  float nanos_per_query = (((double)millis_search) * 1000000.0) / ((double)number_of_positions);
  return nanos_per_query;
}

u64 BenchmarkInfo::get_number_of_bits_padded_bit_vector()
{
  return number_of_bytes_padded_bit_vector * 8ULL;
}

u64 BenchmarkInfo::get_number_of_words_in_bit_vector() { return number_of_words_in_bit_vector; }

u64 BenchmarkInfo::get_number_of_bytes_padded_bit_vector() { return number_of_bytes_padded_bit_vector; }

u64 BenchmarkInfo::get_number_of_words_padded_bit_vector() { return number_of_bytes_padded_bit_vector / 8ULL; }

u64 BenchmarkInfo::get_number_of_bits_padded_layer_0() { return number_of_bytes_padded_layer_0 * 8ULL; }
u64 BenchmarkInfo::get_number_of_bits_padded_layer_12() { return number_of_bytes_padded_layer_12 * 8ULL; }

u64 BenchmarkInfo::get_number_of_bytes_padded_layer_0() { return number_of_bytes_padded_layer_0; }
u64 BenchmarkInfo::get_number_of_bytes_padded_layer_12() { return number_of_bytes_padded_layer_12; }

u64 BenchmarkInfo::get_number_of_words_padded_layer_0() { return number_of_bytes_padded_layer_0 / 8ULL; }
u64 BenchmarkInfo::get_number_of_words_padded_layer_12() { return number_of_bytes_padded_layer_12 / 8ULL; }
