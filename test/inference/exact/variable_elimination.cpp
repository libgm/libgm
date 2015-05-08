#define BOOST_TEST_MODULE variable_elimination
#include <boost/test/unit_test.hpp>

#include <libgm/inference/exact/variable_elimination.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/diagonal_table_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/random/functional.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include "../../factor/predicates.hpp"

#include <random>

BOOST_AUTO_TEST_CASE(test_grid) {
  using namespace libgm;

  // create the variables
  universe u;
  std::size_t m = 4;
  std::size_t n = 3;
  domain variables = u.new_finite_variables(m * n, "v", 2);

  // generate a random Markov network
  pairwise_markov_network<ptable> mn;
  make_grid_graph(variables, m, n, mn);
  std::mt19937 rng;
  mn.initialize(marginal_fn(uniform_table_generator<ptable>(), rng),
                marginal_fn(diagonal_table_generator<ptable>(), rng));

  // for each edge, verify that the marginal over this edge
  // computed directly matches the one computed by variable elimination
  ptable product = prod_all(mn);
  for (auto e : mn.edges()) {
    domain retain({e.source(), e.target()});
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, retain, sum_product<ptable>());
    ptable elim_result = ptable(retain, 1.0) * prod_all(factors);
    ptable direct_result = product.marginal(retain);
    BOOST_CHECK(are_close(elim_result, direct_result, 1e-3));
  }
}
