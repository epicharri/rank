#pragma once
#include "../include/benchmark_info.hpp"
#include "../include/enums.hpp"
#include "../include/globals.hpp"
#include <regex>
#include <string>
#include <vector>

namespace epic
{

    struct
    {
        std::string do_not_store_results = "--do-not-store-results";
        std::string with_shuffles = "--with-shuffles";
        std::string sequential_positions = "--sequential-positions";
        std::string random_positions = "--random-positions";
        std::regex start_position{"^--start-position=([0-9]+)$"};
        std::regex bits_in_superblock_regex{"^--bits-in-superblock=(256|512|1024|2048|4096)$"};
        std::regex bits_in_bit_vector{"^--bits-in-bit-vector=([0-9]+)$"};
        std::string random_bit_vector = "--random-bit-vector";
        std::string one_zero_and_then_all_ones_bit_vector = "--one-zero-and-then-all-ones-bit-vector";
        std::regex query_positions_count{"^--query-positions-count=([0-9]+)$"};
        std::regex device_set_limit_presearch_regex{"^--device-set-limit-presearch=(32|64|128)$"};
        std::regex device_set_limit_search_regex{"^--device-set-limit-search=(32|64|128)$"};
        std::regex threads_per_block_regex{"^--threads-per-block=([0-9][0-9][0-9]?[0-9]?)$"};
        std::regex rank_structure_poppy_regex{"^--rank-structure=poppy$"};
        std::regex rank_structure_cum_poppy_regex{"^--rank-structure=cum-poppy$"};
    } Options;

    struct Parameters
    {
        u64 bits_in_bit_vector = 0ULL;
        int bit_vector_data_type = epic::kind::one_zero_and_then_all_ones_bit_vector; // Default
        u64 query_positions_count = 0ULL;
        u64 start_position = 0ULL;
        int positions_type = epic::kind::random_positions;
        u32 bits_in_superblock = 1024U; // Default
        int device_set_limit_presearch = 64;
        int device_set_limit_search = 64;
        bool with_shuffles = false;
        bool store_results = true;
        int rank_structure_version = kind::poppy; // Default
        int rank_data_word_size = 64;             // Default
        u32 threads_per_block = 0U;
        void print_copyright();
        void print_help();
        int read_arguments(int argc, char **argv, cudaDeviceProp &);
        BenchmarkInfo benchmark_info;
        Parameters() = default;
    };

    void Parameters::print_copyright()
    {
#ifdef BENCHMARK
        fprintf(stderr, "\n------------------------------\n");
#else
        fprintf(stderr, "\nRank function in GPU, Copyright (C) 2023  Harri Kähkönen\nThis program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation. See https://github.com/epicharri/rank/LICENSE.txt.\n\n");
#endif
    }

    void Parameters::print_help()
    {
        fprintf(stderr, "Here will be printing of the instructions.\n");
    }

    int Parameters::read_arguments(int argc, char **argv, cudaDeviceProp &prop)
    {
        cudaGetDeviceProperties(&prop, 0);
        u32 max_threads_per_block = prop.maxThreadsPerBlock;
        print_copyright();
        std::vector<std::string> arguments;

        for (int i = 1; i < argc; i++)
        {
            arguments.push_back(std::string(argv[i]));
        }
        if (argc == 1)
        {
            print_help();
            return 1;
        }
        if (arguments[0] == "help")
        {
            print_help();
            return 1;
        }

        for (int i = 0; i < arguments.size(); i++)
        {
            std::string parameter = arguments[i];
            if (parameter.compare(Options.do_not_store_results) == 0)
            { // == 0 means equality
                store_results = false;
                continue;
            }

            if (parameter.compare(Options.with_shuffles) == 0)
            { // == 0 means equality
                with_shuffles = true;
                continue;
            }

            if (parameter.compare(Options.sequential_positions) == 0)
            { // == 0 means equality
                positions_type = epic::kind::sequential_positions;
                continue;
            }

            if (parameter.compare(Options.random_positions) == 0)
            { // == 0 means equality
                positions_type = epic::kind::random_positions;
                continue;
            }

            if (std::regex_match(parameter, Options.rank_structure_poppy_regex))
            {
                rank_structure_version = epic::kind::poppy;
                continue;
            }
            if (std::regex_match(parameter, Options.rank_structure_cum_poppy_regex))
            {
                rank_structure_version = epic::kind::cum_poppy;
                continue;
            }
            if (std::regex_match(parameter, Options.bits_in_superblock_regex))
            {
                bits_in_superblock = std::stoi(parameter.substr(21, 4));
                continue;
            }
            if (std::regex_match(parameter, Options.device_set_limit_presearch_regex))
            {
                device_set_limit_presearch = std::stoi(parameter.substr(29, 3));
                continue;
            }
            if (std::regex_match(parameter, Options.device_set_limit_search_regex))
            {
                device_set_limit_search = std::stoi(parameter.substr(26, 3));
                continue;
            }

            if (std::regex_match(parameter, Options.bits_in_bit_vector))
            {
                bits_in_bit_vector = std::stoull(parameter.substr(21, parameter.length()));
            }

            if (std::regex_match(parameter, Options.start_position))
            {
                start_position = std::stoull(parameter.substr(17, parameter.length()));
            }

            if (parameter.compare(Options.random_bit_vector) == 0)
            { // == 0 means equality
                bit_vector_data_type = epic::kind::random_bit_vector;
                continue;
            }

            if (parameter.compare(Options.one_zero_and_then_all_ones_bit_vector) == 0)
            { // == 0 means equality
                bit_vector_data_type = epic::kind::one_zero_and_then_all_ones_bit_vector;
                continue;
            }

            if (std::regex_match(parameter, Options.query_positions_count))
            {
                query_positions_count = std::stoull(parameter.substr(24, parameter.length()));
            }
            if (std::regex_match(parameter, Options.threads_per_block_regex))
            {
                u32 par_threads_per_block = (u32)std::stoi(parameter.substr(20, 4));
                if ((par_threads_per_block < 32) || (par_threads_per_block > max_threads_per_block) || (par_threads_per_block % 32))
                {
                    fprintf(stderr, "Number of threads per block should have been at least 32 and at most %" PRIu32 ". We use now the maximum number of threads per block.\n", max_threads_per_block);
                }
                else
                {
                    threads_per_block = par_threads_per_block;
                }
                continue;
            }
        }
        if (store_results)
        {
            benchmark_info.store_results_info = "The results are stored"
        }
        else
        {
            benchmark_info.store_results_info = "The results are not stored"
        }
        if (threads_per_block == 0U)
            threads_per_block = max_threads_per_block;
        benchmark_info.threads_per_block = threads_per_block;
        std::string version = "";
        if (rank_structure_version == kind::poppy)
            version = "poppy";
        if (rank_structure_version == kind::cum_poppy)
            version = "cum-poppy";
        benchmark_info.rank_structure_version = version;
        benchmark_info.number_of_bits_in_bit_vector = bits_in_bit_vector;
        fprintf(stderr, "Bits in the bit vector = %" PRIu64 "\n", bits_in_bit_vector);
        if (positions_type == epic::kind::sequential_positions)
        {
            fprintf(stderr, "Start position = %" PRIu64 "\n", start_position);
            benchmark_info.positions_type = "sequential positions";
            benchmark_info.start_position = start_position;
        }
        if (positions_type == epic::kind::random_positions)
        {
            benchmark_info.positions_type = "random positions";
        }
        benchmark_info.number_of_positions = query_positions_count;
        fprintf(stderr, "Number of positions = %" PRIu64 "\n", query_positions_count);

        std::string bit_vector_content = "Bit vector content: after one zero, all bits are ones.";
        if (bit_vector_data_type == epic::kind::random_bit_vector)
        {
            bit_vector_content = "Bit vector content: randomly generated bits.";
            benchmark_info.bit_vector_content = "random bit vector";
        }
        if (bit_vector_data_type == epic::kind::one_zero_and_then_all_ones_bit_vector)
        {
            benchmark_info.bit_vector_content = "one zero and the rest are ones";
        }
        benchmark_info.bits_in_superblock = bits_in_superblock;
        fprintf(stderr, "%s\n", bit_vector_content.data());

        fprintf(stderr, "Rank version: %s\n", version.c_str());
        fprintf(stderr, "Bits in superblock = %d.\n", bits_in_superblock);
        if (with_shuffles &&
            (bits_in_superblock == 1024 || bits_in_superblock == 2048))
        {
            benchmark_info.shuffles = "With shuffles";
            fprintf(stderr, "With shuffles.\n");
        }
        else
        {
            benchmark_info.shuffles = "Without shuffles";
            fprintf(stderr, "Without shuffles.\n");
        }
        fprintf(stderr, "Set LimitMaxL2FetchGranularity in presearch to %d.\n", device_set_limit_presearch);
        fprintf(stderr, "Set LimitMaxL2FetchGranularity in search to %d.\n\n", device_set_limit_search);
        DEBUG_CODE(fprintf(stderr, "Threads per block: %" PRIu32 ".\n", threads_per_block);)
        if ((bits_in_superblock == 4096) && (rank_structure_version != kind::poppy))
        {
            fprintf(stderr, "Superblock size 4096 bits is supported only in the rank structure version poppy. Please choose some other parameters.\n");
            return 1;
        }
        DEBUG_CODE(

            fprintf(stderr, "Device name: %s\n", prop.name);

            DEBUG_CODE(
                fprintf(stderr, "Device Number: %d\n", 0);
                fprintf(stderr, "  Device name: %s\n", prop.name);
                fprintf(stderr, "  multiProcessorCount: %d\n", prop.multiProcessorCount);
                fprintf(stderr, "  maxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
                fprintf(stderr, "  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
                fprintf(stderr, "  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
                fprintf(stderr, "  maxGridSize[0]: %d\n", prop.maxGridSize[0]);
                fprintf(stderr, "  maxGridSize[1]: %d\n", prop.maxGridSize[1]);
                fprintf(stderr, "  maxGridSize[2]: %d\n", prop.maxGridSize[2]);
                fprintf(stderr, "  maxThreadsDim[0]: %d\n", prop.maxThreadsDim[0]);
                fprintf(stderr, "  maxThreadsDim[1]: %d\n", prop.maxThreadsDim[1]);
                fprintf(stderr, "  maxThreadsDim[2]: %d\n", prop.maxThreadsDim[2]);))

        return 0;
    }

}
