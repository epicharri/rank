#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"

struct DeviceArray
{
  u64 *data = nullptr;
  u64 size_in_bytes = 0ULL;
  int create(u64);
  DeviceArray() = default;
  ~DeviceArray();
};

DeviceArray::~DeviceArray()
{
  if (data)
  {
    CHECK_WITHOUT_RETURN(cudaFree(data));
  }
}

int DeviceArray::create(u64 the_size_in_bytes)
{
  size_in_bytes = the_size_in_bytes;
  CHECK(cudaMalloc((void **)&data, the_size_in_bytes));
  return 0;
}
