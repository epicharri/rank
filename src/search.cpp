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
    fprintf(stderr, "Something went wrong in searcher.create().\n");
    return 0;
  }
  if (searcher.search())
  {
    fprintf(stderr, "Something went wrong in searcher.search().\n");
    return 0;
  }
  if (searcher.fetch_results())
  {
    fprintf(stderr, "Something went wrong in searcher.fetch_results().\n");
    return 0;
  }
  if (searcher.check())
  {
    fprintf(stderr, "Something went wrong in searcher.check().\n");
    return 0;
  }
  if (searcher.print_results())
  {
    fprintf(stderr, "Something went wrong in searcher.print_results().\n");
    return 0;
  }
  return 0;
}
