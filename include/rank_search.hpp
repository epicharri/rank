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
  HostArray host_positions_in;
  HostArray host_results_out;
  DeviceArray device_positions_in_and_results_out;
  u64 number_of_positions = 0ULL;
  bool device_is_nvidia_a100 = true;
  int save_results(u64);
  int fetch_results();
  int check();
  int create();
  int search();
  inline u64 give_random_position(u64);
  int create_random_positions();
  int create_sequential_positions(u64);
  int print_benchmark_info();
  RankSearch() = default;
  ~RankSearch();
};

RankSearch::~RankSearch()
{
}

int RankSearch::print_benchmark_info()
{
  return parameters.benchmark_info.print_info();
}

int RankSearch::fetch_results()
{
  float millis = 0.0;
  device_stream.start_timer();
  cudaMemcpyAsync(host_results_out.data, device_positions_in_and_results_out.data, host_results_out.size_in_bytes, cudaMemcpyDeviceToHost, device_stream.stream);
  device_stream.stop_timer();
  millis = device_stream.duration_in_millis();
  parameters.benchmark_info.millis_transfer_results_D_to_H = millis;
  DEBUG_CODE(fprintf(stderr, "Fetch results from device to host: %f ms\n", millis);
             epic::gpu::get_and_print_last_error("After cudaMemcpy in rank_search.hpp fetch_results() ");)
  return 0;
}

int RankSearch::save_results(u64 count = 0ULL)
{
  if (!parameters.store_results)
    return 0;
  for (u64 i = 0ULL; i < number_of_positions && i < number_of_positions + count; i += 1ULL)
  {
    fprintf(stdout, "%" PRIu64 " ", host_results_out.data[i]);
  }
  return 0;
}

int RankSearch::check()
{
  if (parameters.bit_vector_data_type == epic::kind::one_zero_and_then_all_ones_bit_vector)
  {
    u64 number_of_errors = 0ULL;
    u64 position, rank;
    u64 first_error_position = 0xffff'ffff'ffff'ffff;
    u64 first_error_rank = 0xffff'ffff'ffff'ffff;
    u64 first_error_index = 0xffff'ffff'ffff'ffff;
    bool first_error_not_found = true;
    u64 err;
    for (u64 i = 0ULL; i < number_of_positions; i += 1ULL)
    {
      position = host_positions_in.data[i]; // rank(0) = 0, rank (1) = 0, rank (2) = 1, rank(3) = 2, ...
      rank = host_results_out.data[i];

      if (position == 0ULL)
      {
        err = (u64)(rank != 0ULL);
        number_of_errors += err;
        if (err && first_error_not_found)
        {
          first_error_position = 0ULL;
          first_error_rank = rank;
          first_error_index = 0ULL;
          first_error_not_found = false;
        }
      }
      else
      {
        err = (u64)(rank != (position - 1ULL));
        number_of_errors += err;
        if (err && first_error_not_found)
        {
          first_error_position = position;
          first_error_rank = rank;
          first_error_index = i;
          first_error_not_found = false;
        }
      }
    }
    parameters.benchmark_info.number_of_errors = number_of_errors;
    DEBUG_CODE(
        if (number_of_errors) {
          fprintf(stderr, "ERROR!!! Number of errors is %" PRIu64 "\n", number_of_errors);
          fprintf(stderr, "First error position: %" PRIu64 "\n", first_error_position);
          fprintf(stderr, "First error rank: %" PRIu64 "\n", first_error_rank);
          fprintf(stderr, "First error index: %" PRIu64 "\n", first_error_index);
        } else {
          fprintf(stderr, "SUCCESS!!! The rank function returns correct values.\n");
        });
  }
  else
  {
    fprintf(stderr, "The checking of the results is only for the bit vector content with one zero and the rest all ones.\n");
  }
  return 0;
}

int RankSearch::search()
{
  DEBUG_CODE(fprintf(stderr, "Starting the search.\n"););
  float millis = 22222.0;
  millis = call_rank_search(parameters, device_stream, bit_vector, bit_vector.rank_index, device_positions_in_and_results_out, number_of_positions, device_is_nvidia_a100);
  parameters.benchmark_info.millis_search = millis;
  DEBUG_CODE(
      fprintf(stderr, "Search took %f ms.\n", millis);
      float nanos_per_query = (((double)millis) * 1000000.0) / ((double)number_of_positions);
      fprintf(stderr, "Search per query %f ns.\n", nanos_per_query);
      epic::gpu::get_and_print_last_error("After calling call_rank_search() in rank_search.hpp ");)
  return 0;
}

int RankSearch::create()
{
  device_stream.create();

  device_stream.start_timer();
  auto start = START_TIME;
  int created = bit_vector.create(parameters, device_stream);
  int constructed = bit_vector.construct(parameters, device_stream);
  device_stream.stop_timer();
  float millis_stream = device_stream.duration_in_millis(); // This synchronizes the stream, i.e. blocks CPU until ready.
  device_stream.start_timer();
  if (bit_vector.destruct_host_data())
    return 1;
  device_stream.stop_timer();
  parameters.benchmark_info.millis_free_host_memory_of_bit_vector = device_stream.duration_in_millis();
  auto stop = STOP_TIME;
  float millis = DURATION_IN_MILLISECONDS(start, stop);
  DEBUG_CODE(

      fprintf(stderr, "GPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU takes %f ms.\n", millis_stream);
      fprintf(stderr, "CPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU, including destruction of the host array of the bit vector takes %f ms.\n", millis);)

  number_of_positions = parameters.query_positions_count;

  device_stream.start_timer();
  auto start_create_positions = START_TIME;
  host_positions_in.create(number_of_positions * sizeof(u64), epic::kind::not_write_only); // This will be written and read.
  host_results_out.create(number_of_positions * sizeof(u64), epic::kind::not_write_only);  // This will be written and read.
  device_positions_in_and_results_out.create(number_of_positions * sizeof(u64), device_stream);

  DEBUG_CODE(
      fprintf(stderr, "Number of bits in the bitvector is %" PRIu64 "\n", bit_vector.number_of_bits);
      fprintf(stderr, "Size of the bit vector data array is %" PRIu64 " bytes.\n", bit_vector.device_data.size_in_bytes);

      fprintf(stderr, "Number of positions is %" PRIu64 "\n", number_of_positions);
      fprintf(stderr, "Size of the positions array is %" PRIu64 " bytes\n", host_positions_in.size_in_bytes);

  )

  if (parameters.positions_type == epic::kind::random_positions)
  {
    if (create_random_positions())
    {
      DEBUG_CODE(fprintf(stderr, "Creating random positions did not succeed.\n");)
      return 1;
    }
    DEBUG_CODE(fprintf(stderr, "Creating random positions: SUCCESS.\n");)
  }
  if (parameters.positions_type == epic::kind::sequential_positions)
  {
    DEBUG_CODE(fprintf(stderr, "Start position: %" PRIu64 "\n", parameters.start_position);)

    if (create_sequential_positions(parameters.start_position))
    {
      DEBUG_CODE(fprintf(stderr, "Creating sequential positions did not succeed.\n");)
      return 1;
    }
    DEBUG_CODE(fprintf(stderr, "Creating sequential positions: SUCCESS.\n");)
  }

  device_stream.stop_timer();
  float millis_stream_create_positions = device_stream.duration_in_millis(); // This synchronizes the stream, i.e. blocks CPU until ready.
  auto stop_create_positions = STOP_TIME;
  float millis_create_positions = DURATION_IN_MILLISECONDS(start_create_positions, stop_create_positions);
  DEBUG_CODE(fprintf(stderr, "GPU-timer: Creating the random positions and transfer to GPU takes %f ms.\n", millis_stream_create_positions);)

  DEBUG_CODE(fprintf(stderr, "CPU-timer: Creating the random positions and transfer to GPU takes %f ms.\n", millis_create_positions);)

  return 0;
}

int RankSearch::create_sequential_positions(u64 start)
{
  u64 position;
  for (u64 j = 0ULL; j < number_of_positions; j += 1ULL)
  {
    position = start + j;
    host_positions_in.data[j] = position;
  }
  device_stream.start_timer();
  CHECK(cudaMemcpyAsync(device_positions_in_and_results_out.data, host_positions_in.data, host_positions_in.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))
  device_stream.stop_timer();
  parameters.benchmark_info.millis_transfer_positions_H_to_D = device_stream.duration_in_millis();
  return 0;
}

inline u64 RankSearch::give_random_position(u64 position_index)
{
  u64 ls_31_bits = (u64)random();
  u64 ms_bits = ((u64)random()) << 31;
  u64 position = (ms_bits | ls_31_bits) % parameters.bits_in_bit_vector;
  return (position_index != number_of_positions - 1ULL) ? position : number_of_positions; // If position_index is the last one, return number_of_positions. This is to test that also rank(number_of_positions) works.
}

int RankSearch::create_random_positions()
{
  srandom(2);
  u64 position;
  for (u64 j = 0ULL; j < number_of_positions; j += 1ULL)
  {
    position = give_random_position(j);
    host_positions_in.data[j] = position;
  }
  device_stream.start_timer();
  CHECK(cudaMemcpyAsync(device_positions_in_and_results_out.data, host_positions_in.data, host_positions_in.size_in_bytes, cudaMemcpyHostToDevice, device_stream.stream))
  device_stream.stop_timer();
  parameters.benchmark_info.millis_transfer_positions_H_to_D = device_stream.duration_in_millis();
  return 0;
}