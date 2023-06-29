#pragma once
#include "../include/globals.hpp"
#include "../include/gpu/cuda.hpp"

struct HostArray
{
  u64 *data = nullptr;
  u64 size_in_bytes = 0ULL;
  int create(u64, bool);
  int destruct();
  HostArray() = default;
  ~HostArray();
};

HostArray::~HostArray()
{
  if (data)
  {
    CHECK_WITHOUT_RETURN(cudaFreeHost(data));
  }
};

int HostArray::destruct()
{
  if (data)
  {
    CHECK_WITHOUT_RETURN(cudaFreeHost(data));
    DEBUG_CODE(fprintf(stderr, "HostArray data %" PRIu64 " bytes freed.\n", size_in_bytes);)
  }
  return 0;
};

int HostArray::create(u64 the_size_in_bytes, bool write_only)
{
  size_in_bytes = the_size_in_bytes;
  if (write_only)
  {
    // Faster if write-only by host. Does not fill CPU cache.
    CHECK(cudaHostAlloc((void **)&data, the_size_in_bytes, cudaHostAllocWriteCombined));
  }
  else
  {
    CHECK(cudaHostAlloc((void **)&data, the_size_in_bytes, cudaHostAllocDefault));
  }
  return 0;
}
