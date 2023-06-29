#pragma once
#include "../include/bit_vector.hpp"
#include "../include/call_rank_search.hpp"
#include "../include/device_array.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_stream.hpp"
#include "../include/host_array.hpp"
#include "../include/parameters.hpp"
#include "../include/utils/helpers.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>

class RankSearch
{

public:
  epic::Parameters parameters;
  epic::gpu::DeviceStream device_stream;
  BitVector bit_vector;
  HostArray host_positions_in_and_results_out;
  DeviceArray device_positions_in_and_results_out;
  bool device_is_nvidia_a100 = true;
  int create();
  int search();
  inline u64 give_random_position(u64, u64);
  int create_random_positions();
  RankSearch() = default;
  ~RankSearch();
};

RankSearch::~RankSearch()
{
}

int RankSearch::search()
{
  fprintf(stderr, "Starting the search.\n");
  u64 number_of_positions = parameters.query_positions_count;
  float millis = call_rank_search(parameters, device_stream, bit_vector, bit_vector.rank_index, device_positions_in_and_results_out, number_of_positions, device_is_nvidia_a100);
  epic::gpu::get_and_print_last_error();
  fprintf(stderr, "Search took %f ms.\n");
  float nanos_per_query = (((double)millis) * 1000000) / ((double)number_of_positions);
  fprintf(stderr, "Search per query %f ns.\n", nanos_per_query);
}

int RankSearch::create()
{
  device_stream.create();

  device_stream.start_timer();
  auto start = START_TIME;
  int created = bit_vector.create(parameters, device_stream);
  int constructed = bit_vector.construct(device_stream);
  device_stream.stop_timer();
  float millis_stream = device_stream.duration_in_millis(); // This synchronizes the stream, i.e. blocks CPU until ready.
  if (bit_vector.destruct_host_data())
    return 1;
  auto stop = STOP_TIME;
  float millis = DURATION_IN_MILLISECONDS(start, stop);
  BENCHMARK_CODE(

      fprintf(stderr, "GPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU takes %f ms.\n", millis_stream);
      fprintf(stderr, "CPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU, including destruction of the host array of the bit vector takes %f ms.\n", millis);)

  u64 number_of_positions = parameters.query_positions_count;

  device_stream.start_timer();
  auto start_create_positions = START_TIME;
  host_positions_in_and_results_out.create(number_of_positions * sizeof(u64), epic::kind::not_write_only); // This will be written and read.
  device_positions_in_and_results_out.create(number_of_positions * sizeof(u64), device_stream);

  BENCHMARK_CODE(
      fprintf(stderr, "Number of bits in the bitvector is %" PRIu64 "\n", bit_vector.number_of_bits);
      fprintf(stderr, "Size of the bit vector data array is %" PRIu64 " bytes.\n", bit_vector.device_data.size_in_bytes);

      fprintf(stderr, "Number of positions is %" PRIu64 "\n", number_of_positions);
      fprintf(stderr, "Size of the positions array is %" PRIu64 " bytes\n", host_positions_in_and_results_out.size_in_bytes);

  )

  if (create_random_positions())
  {
    DEBUG_CODE(fprintf(stderr, "Creating random positions did not succeed.\n");)
    return 1;
  }
  device_stream.stop_timer();
  float millis_stream_create_positions = device_stream.duration_in_millis(); // This synchronizes the stream, i.e. blocks CPU until ready.
  auto stop_create_positions = STOP_TIME;
  float millis_create_positions = DURATION_IN_MILLISECONDS(start_create_positions, stop_create_positions);
  BENCHMARK_CODE(fprintf(stderr, "GPU-timer: Creating the random positions and transfer to GPU takes %f ms.\n", millis_stream_create_positions);)

  BENCHMARK_CODE(fprintf(stderr, "CPU-timer: Creating the random positions and transfer to GPU takes %f ms.\n", millis_create_positions);)

  return 0;
}

inline u64 RankSearch::give_random_position(u64 number_of_positions, u64 position_index)
{
  u64 ls_31_bits = (u64)random();
  u64 ms_bits = ((u64)random()) << 31;
  u64 position = (ms_bits | ls_31_bits) % parameters.bits_in_bit_vector;
  return (position_index != number_of_positions - 1ULL) ? position : number_of_positions; // If position_index is the last one, return number_of_positions. This is to test that also rank(number_of_positions) works.
}

int RankSearch::create_random_positions()
{
  u64 number_of_positions = parameters.query_positions_count;
  u64 batch_size_in_bytes = 1ULL << 30; // 2**30 B = 1 GB
  u64 batch_size_in_words = batch_size_in_bytes / sizeof(u64);
  u64 last_batch_number = number_of_positions / batch_size_in_words;

  if (number_of_positions < batch_size_in_words)
  {
    batch_size_in_words = number_of_positions;
    batch_size_in_bytes = batch_size_in_words * sizeof(u64);
    last_batch_number = 0ULL;
  }

  srandom(2);

  u64 position_index = 0ULL;
  for (u64 batch_number = 0ULL; batch_number < last_batch_number; batch_number += 1ULL)
  {
    for (u64 j = 0ULL; j < batch_size_in_words; j += 1ULL)
    {
      position_index = batch_number * batch_size_in_words + j;
      host_positions_in_and_results_out.data[position_index] = give_random_position(number_of_positions, position_index);
    }
    CHECK(cudaMemcpyAsync(device_positions_in_and_results_out.data + batch_number * batch_size_in_words, host_positions_in_and_results_out.data + batch_number * batch_size_in_words, batch_size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))
    batch_number += 1ULL;
  }

  for (u64 j = last_batch_number * batch_size_in_words; j < number_of_positions; j += 1ULL)
  {
    position_index = j;
    host_positions_in_and_results_out.data[position_index] = give_random_position(number_of_positions, position_index);
  }
  batch_size_in_bytes = (number_of_positions - last_batch_number * batch_size_in_words) * sizeof(u64);
  CHECK(cudaMemcpyAsync(device_positions_in_and_results_out.data + last_batch_number * batch_size_in_words, host_positions_in_and_results_out.data + last_batch_number * batch_size_in_words, batch_size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))

  return 0;
}
