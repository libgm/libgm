#define BOOST_TEST_MODULE variable_elimination
#include <boost/test/unit_test.hpp>

#include <libgm/inference/exact/variable_elimination.hpp>

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/diagonal_table_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

namespace libgm {
  template <>
  struct argument_traits<std::pair<int, int> >
    : fixed_discrete_traits<std::pair<int, int>, 2> { };
}

BOOST_AUTO_TEST_CASE(test_grid) {
  using namespace libgm;
  typedef std::pair<int, int> grid_vertex;
  typedef probability_table<grid_vertex> ptable;

  // generate a Markov network with random potentials
  std::size_t m = 4;
  std::size_t n = 3;
  std::mt19937 rng;
  pairwise_markov_network<ptable> mn;
  make_grid_graph(m, n, mn);
  mn.initialize(bind_marginal(uniform_table_generator<ptable>(), rng),
                bind_marginal(diagonal_table_generator<ptable>(), rng));

  // for each edge, verify that the marginal over this edge
  // computed directly matches the one computed by variable elimination
  ptable product = prod_all(mn);
  for (undirected_edge<grid_vertex> e : mn.edges()) {
    domain<grid_vertex> retain({e.source(), e.target()});
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, retain, sum_product<ptable>());
    ptable elim_result = prod_all(factors).marginal(retain);
    ptable direct_result = product.marginal(retain);
    BOOST_CHECK_SMALL(max_diff(elim_result, direct_result), 1e-3);
  }
}
