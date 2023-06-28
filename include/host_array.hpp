#pragma once
#include "../include/cuda.hpp"
#include "../include/globals.hpp"
#include <omp.h>

<template typename T = u64> struct HostArray
{
  T *data = nullptr;
  u64 size_in_bytes = 0ULL;
  int create(u64, bool);
  HostArray() = default;
  ~HostArray();
};

HostArray::~HostArray()
{
  if (data)
  {
    CHECK(cudaFreeHost(data));
  }
};

int HostArray::create(u64 the_size_in_bytes, bool write_only)
{
  size_in_bytes = the_size_in_bytes;
  if (write_only)
  {
    // Faster if write-only by host. Does not fill CPU cache.
    CHECK(cudaHostAlloc((void **)&data, the_size_in_bytes, cudaHostAllocWriteCombined););
  }
  else
  {
    CHECK(cudaHostAlloc((void **)&data, the_size_in_bytes, cudaHostAllocDefault););
  }
  return 0;
}
