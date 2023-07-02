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
  {
    fprintf("Something went wronk in searcher.create().\n");
    return 0;
  }
  if (searcher.search())
  {
    fprintf("Something went wronk in searcher.search().\n");
    return 0;
  }
  if (searcher.fetch_results())
  {
    fprintf("Something went wronk in searcher.fetch_results().\n");
    return 0;
  }
  if (searcher.check())
  {
    fprintf("Something went wronk in searcher.check().\n");
    return 0;
  }
  if (searcher.print_results(100000))
  {
    fprintf("Something went wronk in searcher.print_results().\n");
    return 0;
  }
  return 0;
}
