// #include "../include/search.hpp"
#include "../include/globals.hpp"
#include "../include/parameters.hpp"
#include "../include/rank_search.hpp"

int main(int argc, char **argv)
{

  cudaDeviceProp prop;
  epic::Parameters parameters;
  if (parameters.read_arguments(argc, argv, prop))
    return 0;
  RankSearch searcher(parameters, DEVICE_IS_NVIDIA_A100);
  if (searcher.create())
    return 0;

  if (searcher.search())
    return 0;

  return 0;
}
