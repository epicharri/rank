#include "../include/globals.hpp"
#include "../include/parameters.hpp"
#include "../include/rank_search.hpp"

int main(int argc, char **argv)
{

  cudaDeviceProp prop;
  RankSearch searcher;
  if (searcher.parameters.read_arguments(argc, argv, prop))
    return 0;
  if (searcher.create())
    return 0;

  if (searcher.search())
    return 0;

  if (searcher.fetch_results())
    return 0;
  if (searcher.print_results(100000))
    return 0;

  return 0;
}
