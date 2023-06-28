#pragma once
#include "../globals.hpp"
#include "cuda.hpp"
#include "device_stream.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

namespace epic
{
  namespace gpu
  {

    struct DeviceMemory
    {
      u64 *data = nullptr;
      u64 *L0 = nullptr;
      u64 *L12 = nullptr;
      u64 size_data = 0ULL; // Sizes are in bytes;
      DeviceStream device_stream;

      void set_size_of_bit_vector(u64);
      void set_sizes_of_rank_data_structures(u64, u64); // size of L0, size of L12

      int allocate_device_memory();

      DeviceMemory();
      ~DeviceMemory();
    };

    DeviceMemory::DeviceMemory()
    {
      device_stream.create();
    }

    DeviceMemory::~DeviceMemory()
    {
      DEBUG_BEFORE_DESTRUCT("DeviceMemory (ALL)");
      if (data)
        cudaFree(data);
      if (L0)
        cudaFree(L0);
      if (L12)
        cudaFree(L12);
      DEBUG_AFTER_DESTRUCT("DeviceMemory (ALL)");
    }

    // Number of uint64_t words as a parameter.
    void DeviceMemory::set_size_of_bit_vector(u64 number_of_words)
    {
      size_data = (number_of_words * sizeof(u64));
    }

    // Number of uint64_t words of L0 and L12 as parameters.
    void DeviceMemory::set_sizes_of_rank_data_structures(u64 number_of_words_L0, u64 number_of_words_L12)
    {
      size_L0 = (number_of_words_L0 * sizeof(u64));
      size_L12 = (number_of_words_L12 * sizeof(u64));
    }

    int DeviceMemory::allocate_device_memory()
    {
      cudaError_t errors[3];

      errors[0] = deviceMalloc((void **)&data, size_data);
      errors[1] = deviceMalloc((void **)&L0, size_L0);
      errors[2] = deviceMalloc((void **)&L12, size_L12);

      int number_of_errors = 0;
      for (int i = 0; i < 3; i++)
      {
        if (errors[i])
        {
          DEBUG_CODE(print_device_error(errors[i], "");)
          number_of_errors++;
        }
      }
      if (number_of_errors)
        fprintf(stderr, "Allocating device memory failed.\n");
      return number_of_errors;
    }

  }
}