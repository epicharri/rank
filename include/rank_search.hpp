#pragma once
#include "../include/bit_vector.hpp"
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/device_stream.hpp"
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
  bool device_is_nvidia_a100 = true;
  int create();
  int search();
  RankSearch() = default;
  ~RankSearch();
};

RankSearch::~RankSearch()
{
}

int RankSearch::create()
{
  device_stream.create();
  device_stream.start_timer();
  DEBUG_CODE(fprintf(stderr, "Device_stream timer started\n");)
  auto start = START_TIME;
  int created = bit_vector.create(parameters, device_stream);
  int constructed = bit_vector.construct(device_stream);
  device_stream.stop_timer();
  float millis_stream = device_stream.duration_in_millis(); // This synchronizes the stream, i.e. blocks CPU until ready.
  auto stop = STOP_TIME;

  DEBUG_CODE(fprintf(stderr, "created = %d, constructed = %d\n", created, constructed););
  float millis = DURATION_IN_MILLISECONDS(start, stop);
  BENCHMARK_CODE(fprintf(stderr, "GPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU takes %f ms.\n", millis_stream);)

  BENCHMARK_CODE(fprintf(stderr, "CPU-timer: Creating the bit vector and constructing the rank data structures in CPU and transfer to GPU takes %f ms.\n", millis);)

  return 0;
}

int RankSearch::search()
{
  fprintf(stderr, "RankSearch::search() called.\n");
  return 0;
}

/*
  int RankSearch::search()
  {
    auto start_create_and_search = START_TIME;
    auto start_create = START_TIME;
    auto start = START_TIME;
    std::ifstream fs(filename, std::ios::in | std::ios::ate);
    if (!fs.is_open())
    {
      fprintf(stderr, "The file %s can not be opened.\n", filename.c_str());
      return 1;
    }
    if (read_size(fs))
      return 1;
    if (allocate_memory())
      return 1;
    if (read_file(fs, 0))
      return 1;
    if (add_paddings())
      return 1;
    auto stop = STOP_TIME;
    millis = DURATION_IN_MILLISECONDS(start, stop);

    float millis_allocate_parse_create_positions_search = 0.0;
    auto start_allocate_to_search = START_TIME;

    if (allocate_kmer_data_memory(dm))
      return 1;
    if (parse_kmer_string_in_gpu(dm))
      return 1;
    if (create_positions_in_gpu(dm))
      return 1;
    if (call_search(dm))
      return 1;

    DEBUG_CODE(cudaDeviceSynchronize();)
    auto stop_allocate_to_search = STOP_TIME;
    auto stop_create = STOP_TIME;

    auto start_saving_results = START_TIME;
    if (save_results_to_file(dm))
      return 1;
    auto stop_saving_results = STOP_TIME;
    float millis_saving_results = DURATION_IN_MILLISECONDS(start_saving_results, stop_saving_results);
    fprintf(stderr, "Transfer the results from the device to the host memory and saving the results to file: %f seconds.\n", (millis_saving_results / 1000.0));
    millis_allocate_parse_create_positions_search = DURATION_IN_MILLISECONDS(start_allocate_to_search, stop_allocate_to_search);
    millis_create = DURATION_IN_MILLISECONDS(start_create, stop_create);

    fprintf(stderr, "Transfer kmer data as characters to gpu: %f ms.\n", millis_transfer_kmer_raw_data_to_device);
    fprintf(stderr, "Parsing the raw kmer string to 2-bit-wise k-mer string IN GPU: %f ms.\n", millis_parse_kmer_string_in_gpu);
    fprintf(stderr, "From allocate kmer data memory to the finishing of search (add time to read the results from device to host to get total without I/O): %f ms.\n", millis_allocate_parse_create_positions_search);
    fprintf(stderr, "Number of k-mers queried: %" PRIu64 "\n", number_of_positions);
    fprintf(stderr, "Total: create_and_search, using cpu timer with cudaDeviceSynchronize(): %f ms.\n", millis_create);
    float nanos_per_query = (float)(((((double)millis_create) * (double)1000000.0) / ((double)number_of_positions)));
    fprintf(stderr, "Total per query: %f ns.\n", nanos_per_query);

    return 0;
  }

  int RankSearch::save_results_to_file(epic::gpu::DeviceMemory &dm)
  {
    if (persister.save_results_to_file(start_positions, kmer_data, dm, parameters, kmer_size))
      return 1;
    return 0;
  }

  int RankSearch::call_search(epic::gpu::DeviceMemory &dm)
  {
    epic::SearchCaller search_caller;
    search_caller.search(parameters, dm, kmer_data);

    fprintf(stderr, "Search in GPU: %f ms\n", search_caller.millis);
    float nanos_per_query_in_gpu = (float)(((((double)search_caller.millis) * (double)1000000.0) / ((double)number_of_positions)));
    fprintf(stderr, "Search in GPU per query: %f ns\n", nanos_per_query_in_gpu);

    return 0;
  }

  int RankSearch::create_positions_in_gpu(epic::gpu::DeviceMemory &dm)
  {
    epic::DeviceStartPositions positions_creator;
    if (positions_creator.create_positions(
            dm.device_stream,
            start_positions,
            kmer_data,
            kmer_size))
      return 1;
    millis_create_positions = positions_creator.millis_create_positions;
    return 0;
  }

  int RankSearch::allocate_kmer_data_memory(epic::gpu::DeviceMemory &dm)
  {
    auto start = START_TIME;
    kmer_data.number_of_positions = number_of_positions;
    kmer_data.number_of_positions_padded = epic::utils::round_up_first_to_multiple_of_second<u64>(number_of_positions, 64ULL);
    kmer_data.size_positions_and_results = kmer_data.number_of_positions_padded * sizeof(u64);
    kmer_data.size_kmer_string = number_of_letters_padded / 4ULL;
    kmer_data.number_of_words_in_kmer_string = kmer_data.size_kmer_string / sizeof(u64);
    kmer_data.size_raw_data = number_of_letters_padded;
    if (kmer_data.allocate_memory(dm.device_stream))
      return 1;
    auto stop = STOP_TIME;
    millis_allocate_kmer_data_memory = DURATION_IN_MILLISECONDS(start, stop);
    return 0;
  }

  int RankSearch::parse_kmer_string_par(epic::gpu::DeviceMemory &dm)
  {
    auto start = START_TIME;

    // We assume that the target machine is little-endian.

#pragma omp parallel for
    for (u64 i = 0ULL; i < number_of_letters_padded; i += 32ULL)
    {
      u8 word[8] = {};
#pragma unroll
      for (u32 j = 0U; j < 8U; j += 1U)
      {
        word[7U - j] =
            bytes[bits[raw_data[i + j * 4U]]][0] |
            bytes[bits[raw_data[i + j * 4U + 1U]]][1] |
            bytes[bits[raw_data[i + j * 4U + 2U]]][2] |
            bytes[bits[raw_data[i + j * 4U + 3U]]][3];
      }
      kmer_data.kmer_string[i / 32ULL] = reinterpret_cast<u64 *>(word)[0];
    }

    CHECK(cudaMemcpyAsync(kmer_data.d_kmer_string,
                          kmer_data.kmer_string,
                          kmer_data.size_kmer_string,
                          cudaMemcpyHostToDevice,
                          dm.device_stream.stream));
    auto stop = STOP_TIME;
    millis_parse_kmer_string = DURATION_IN_MILLISECONDS(start, stop);

    return 0;
  }

  int RankSearch::parse_kmer_string(epic::gpu::DeviceMemory &dm)
  {
    auto start = START_TIME;
    // We assume that the target machine is little-endian.
    for (u64 i = 0ULL; i < number_of_letters_padded; i += 32ULL)
    {
      u8 word[8] = {};
#pragma unroll
      for (u32 j = 0U; j < 8U; j += 1U)
      {
        word[7U - j] =
            bytes[bits[raw_data[i + j * 4U]]][0] |
            bytes[bits[raw_data[i + j * 4U + 1U]]][1] |
            bytes[bits[raw_data[i + j * 4U + 2U]]][2] |
            bytes[bits[raw_data[i + j * 4U + 3U]]][3];
      }
      kmer_data.kmer_string[i / 32ULL] = reinterpret_cast<u64 *>(word)[0];
    }
    CHECK(cudaMemcpyAsync(
        kmer_data.d_kmer_string,
        kmer_data.kmer_string,
        kmer_data.size_kmer_string,
        cudaMemcpyHostToDevice,
        dm.device_stream.stream));

    auto stop = STOP_TIME;
    millis_parse_kmer_string = DURATION_IN_MILLISECONDS(start, stop);

    return 0;
  }

  int RankSearch::parse_kmer_string_in_gpu(epic::gpu::DeviceMemory &dm)
  {
    BENCHMARK_CODE(cudaStreamSynchronize(dm.device_stream.stream);) // This synchronization is needed here only for benchmarking the following memcpy. This can be removed.
    dm.device_stream.start_timer();
    CHECK(cudaMemcpyAsync(
        kmer_data.d_raw_data,
        raw_data.data(),
        kmer_data.size_raw_data,
        cudaMemcpyHostToDevice,
        dm.device_stream.stream));
    dm.device_stream.stop_timer();
    millis_transfer_kmer_raw_data_to_device = dm.device_stream.duration_in_millis();
    dm.device_stream.start_timer();
    u64 block_size, grid_size;
    block_size = 256;
    grid_size = epic::utils::round_up_first_to_multiple_of_second<u64>(number_of_letters_padded / 32ULL, block_size) / block_size;
    epic::gpu::create_kmer_string<<<dim3(grid_size), dim3(block_size), 0, dm.device_stream.stream>>>(kmer_data.d_raw_data, kmer_data.d_kmer_string, number_of_letters_padded / 32ULL);
    dm.device_stream.stop_timer();
    millis_parse_kmer_string_in_gpu = dm.device_stream.duration_in_millis();

    epic::gpu::get_and_print_last_error();

    return 0;
  }

  int RankSearch::add_paddings()
  {
    number_of_letters_padded = epic::utils::round_up_first_to_multiple_of_second<u64>((number_of_letters + 32ULL), 32ULL);
    for (u64 i = number_of_letters; i < number_of_letters_padded; i++)
      raw_data[i] = 0;
    raw_data.resize(number_of_letters_padded);
    //  raw_data.shrink_to_fit();
    start_positions.shrink_to_fit();
    return 0;
  }

  inline int RankSearch::allocate_memory()
  {
    try
    {
      raw_data.reserve(size_of_file);
      raw_data.resize(size_of_file);
      start_positions.reserve(size_of_file / assumed_row_length); // If the number of rows is more, the automatic resizing will occur, lowering the performance.
    }
    catch (const std::bad_alloc &e)
    {
      epic::utils::print_host_error(e, "Allocating memory for the query file fails.");
      return 1;
    }
    return 0;
  }

  inline int RankSearch::read_file(std::ifstream &fs, std::size_t pos = 0)
  {
    auto start = START_TIME;
    fs.seekg(pos, std::ios::beg);
    int letters_in_row = 0;
    while (fs.good())
    {
      fs.getline(raw_data.data() + number_of_letters, size_of_file - number_of_letters);
      letters_in_row = fs.gcount();
      letters_in_row = (letters_in_row > 0) * (letters_in_row - 1);
      letters_in_row = (raw_data[number_of_letters] != '>') * letters_in_row;
      if (letters_in_row)
      {
        start_positions.push_back(number_of_letters);
        number_of_rows++;
      }
      number_of_positions += (letters_in_row + 1 - kmer_size) * (letters_in_row >= kmer_size);
      number_of_letters += letters_in_row;
    }
    start_positions.push_back(number_of_letters);
    auto stop = STOP_TIME;
    millis_read_file = DURATION_IN_MILLISECONDS(start, stop);
    return 0;
  }

  inline int RankSearch::read_file_at_once(std::ifstream &fs, std::size_t pos = 0)
  {
    auto start = START_TIME;
    fs.seekg(pos, std::ios::beg);
    fs.read(raw_data.data(), size_of_file);
    auto stop = STOP_TIME;
    float millis_read_file_at_once = DURATION_IN_MILLISECONDS(start, stop);
    if (fs)
    {
      fprintf(stderr, "File %s read to memory in %f ms.\n", filename.c_str(), millis_read_file_at_once);
      return 0;
    }
    fprintf(stderr, "Error in reading a file %s.\n", filename.c_str());
    return 1;
  }

  inline int RankSearch::read_file_at_once_concurrently()
  {
    auto start = START_TIME;
    u64 number_of_parallel_file_reads = 4ULL;
    u64 max_batch_size = size_of_file / number_of_parallel_file_reads;
    u64 batch_sizes[number_of_parallel_file_reads] = {};
    for (int i = 0; i < number_of_parallel_file_reads - 1; i++)
    {
      batch_sizes[i] = max_batch_size;
    }
    batch_sizes[number_of_parallel_file_reads - 1] =
        size_of_file - (number_of_parallel_file_reads - 1) * max_batch_size;
    int errors[number_of_parallel_file_reads] = {};
    for (int i = 0; i < number_of_parallel_file_reads - 1; i++)
    {
      errors[i] = 0;
    }
#pragma omp parallel
    {
#pragma omp for
      for (u64 i = 0ULL; i < number_of_parallel_file_reads; i += 1ULL)
      {
        std::ifstream fs(filename, std::ios::in | std::ios::binary);
        if (!fs.is_open())
        {
          errors[i] = 1;
          fprintf(stderr, "The file %s can not be opened.\n", filename.c_str());
        }
        else
        {
          fs.seekg(i * max_batch_size, std::ios::beg);
          fs.read(raw_data.data() + (i * max_batch_size), batch_sizes[i]);
          if (!fs)
            errors[i] = 1;
          fs.close();
        }
      }
    }
    auto stop = STOP_TIME;
    float millis_read_file_at_once = DURATION_IN_MILLISECONDS(start, stop);
    int error = 0;
    for (int i = 0; i < number_of_parallel_file_reads - 1; i++)
    {
      error += errors[i];
    }
    if (!error)
    {
      fprintf(stderr, "File %s read concurrently to memory in %f ms.\n", filename.c_str(), millis_read_file_at_once);
      return 0;
    }
    fprintf(stderr, "Error in reading a file %s.\n", filename.c_str());
    return 1;
  }

  inline int
  RankSearch::read_size(std::ifstream &fs)
  {
    size_of_file = fs.tellg();
    if (size_of_file == 0)
    {
      fprintf(stderr, "The size of the file %s is too small to contain any data.\n", filename.c_str());
      return 1;
    }
    fs.seekg(0, std::ios::beg);
    return 0;
  }

  bool RankSearch::test_kmer_string(epic::gpu::DeviceMemory &dm)
  {
    u8 bits[256];
    bits['a'] = bits['A'] = 0U;
    bits['c'] = bits['C'] = 1U;
    bits['g'] = bits['G'] = 2U;
    bits['t'] = bits['T'] = 3U;

    bool success = true;
    u64 *kmer_string_from_gpu = nullptr;
    kmer_string_from_gpu = new u64[number_of_letters_padded / 32ULL];
    CHECK(cudaStreamSynchronize(dm.device_stream.stream));
    CHECK(cudaMemcpy(
        kmer_string_from_gpu,
        kmer_data.d_kmer_string,
        kmer_data.size_kmer_string,
        cudaMemcpyDeviceToHost));
    for (u64 i = 0ULL; i < number_of_letters_padded; i += 32ULL)
    {
      u64 word = kmer_string_from_gpu[i / 32ULL];
      for (u64 j = 0ULL; j < 32ULL; j++)
      {
        u8 letter = (u8)((word >> (2 * (31 - j))) & 0b11ULL);
        if (letter != bits[raw_data[i + j]])
          success = false;
      }
    }
    if (kmer_string_from_gpu)
      delete[] kmer_string_from_gpu;
    if (success)
      fprintf(stderr, "SUCCESS:: Creating the kmer string in the gpu produces the correct 2-bit presentations.\n");
    else
      fprintf(stderr, "ERRORS in creating the kmer string in gpu.\n");
    return success;
  }

  void RankSearch::print_performance_info()
  {
    fprintf(stderr, "Opening the file %s succeeded.\n", filename.c_str());
    fprintf(stderr, "Number of bytes in file: %zu \n", size_of_file);
    fprintf(stderr, "Opening the file, allocating memory, and reading the file and parsing into a character array takes %f ms.\n", millis);
    fprintf(stderr, "Just reading the file and parsing into a character array %f ms.\n", millis_read_file);
    fprintf(stderr, "Allocating host memory and async all to allocate device memory, for kmer string and positions: %f ms.\n", millis_allocate_kmer_data_memory);
    fprintf(stderr, "Parsing the raw kmer string to 2-bit-wise k-mer string: %f ms.\n", millis_parse_kmer_string);
    fprintf(stderr, "Parsing the raw kmer string to 2-bit-wise k-mer string IN GPU: %f ms.\n", millis_parse_kmer_string_in_gpu);
    fprintf(stderr, "Allocate device memory for start positions, and create the positions %f ms.\n", millis_create_positions);
    fprintf(stderr, "Total: From opening the fasta file, through kmer parsing, until kmer positions array and kmer string is constructed in the device: %f ms.\n", millis_create);

    fprintf(stderr, "Number of letters %d\n", (int)number_of_letters);
    fprintf(stderr, "Number of rows %d\n", (int)number_of_rows);
    fprintf(stderr, "Number of positions %" PRIu64 "\n", number_of_positions);
  }
}
*/