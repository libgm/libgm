#ifndef LIBGM_TEST_MN_FIXTURE_HPP
#define LIBGM_TEST_MN_FIXTURE_HPP

#include <libgm/argument/universe.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/diagonal_table_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/random/functional.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>

#include <random>

using namespace libgm;

struct fixture {
  fixture() {
    std::size_t m = 5;
    std::size_t n = 4;
    vars = u.new_finite_variables(m * n, "v", 2);
    std::mt19937 rng;
    make_grid_graph(vars, m, n, mn);
    mn.initialize(marginal_fn(uniform_table_generator<ptable>(), rng),
                  marginal_fn(diagonal_table_generator<ptable>(), rng));
  }

  void check_belief(const ptable& belief, double tol) {
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(), sum_product<ptable>());
    ptable expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected), tol);
  }

  void check_belief_normalized(const ptable& belief, double tol) {
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(), sum_product<ptable>());
    ptable expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected.normalize()), tol);
  }

  universe u;
  domain vars;
  pairwise_markov_network<ptable> mn;
};

#endif

