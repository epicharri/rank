#pragma once
#include "../include/globals.hpp"

struct NormalHostArray
{
  u64 *data = nullptr;
  u64 size_in_bytes = 0ULL;
  int create(u64);
  int destruct();
  HostArrNormalHostArrayay() = default;
  ~NormalHostArray();
};

NormalHostArray::~NormalHostArray()
{
  if (data)
  {
    delete[] data;
  }
};

int NormalHostArray::destruct()
{
  if (data)
  {
    delete[] data;
    data = nullptr;
  }
  return 0;
};

int NormalHostArray::create(u64 the_size_in_bytes)
{
  size_in_bytes = the_size_in_bytes;
  data = new u64[the_size_in_bytes];
  return 0;
}
