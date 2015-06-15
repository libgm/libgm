#ifndef LIBGM_TEST_MN_FIXTURE_HPP
#define LIBGM_TEST_MN_FIXTURE_HPP

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/diagonal_table_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>

#include <random>

namespace libgm {
  template <>
  struct argument_traits<std::pair<int, int> >
    : fixed_discrete_traits<std::pair<int, int>, 2> { };
}

using namespace libgm;

struct fixture {
  typedef probability_table<double, std::pair<int, int> > ptable_type;
  fixture() {
    std::size_t m = 5;
    std::size_t n = 4;
    std::mt19937 rng;
    make_grid_graph(m, n, mn);
    mn.initialize(bind_marginal(uniform_table_generator<ptable_type>(), rng),
                  bind_marginal(diagonal_table_generator<ptable_type>(), rng));
  }

  void check_belief(const ptable_type& belief, double tol) {
    std::list<ptable_type> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(),
                         sum_product<ptable_type>());
    ptable_type expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected), tol);
  }

  void check_belief_normalized(const ptable_type& belief, double tol) {
    std::list<ptable_type> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(),
                         sum_product<ptable_type>());
    ptable_type expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected.normalize()), tol);
  }

  pairwise_markov_network<ptable_type> mn;
};

#endif

