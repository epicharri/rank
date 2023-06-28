#pragma once
#include "../include/gpu/cuda.hpp"
#include "../include/gpu/globals.hpp"

<template typename T = u64> struct DeviceArray
{
  T *data = nullptr;
  u64 size_in_bytes = 0ULL;
  int create(u64);
  DeviceArray() = default;
  ~DeviceArray();
};

DeviceArray::~DeviceArray()
{
  if (data)
  {
    CHECK(cudaFree(data));
  }
}

int DeviceArray::create(u64 the_size_in_bytes)
{
  size_in_bytes = the_size_in_bytes;
  CHECK(cudaMalloc((void **)&data, the_size_in_bytes););
  return 0;
}
