#define BOOST_TEST_MODULE generalized_bp
#include <boost/test/unit_test.hpp>

#include <libgm/inference/loopy/generalized_bp.hpp>
#include <libgm/inference/loopy/generalized_bp_pc.hpp>

#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/decomposable.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <iostream>

typedef std::pair<int, int> grid_vertex;

namespace libgm {
  template class asynchronous_generalized_bp<probability_table<var> >;
  template class asynchronous_generalized_bp_pc<probability_table<var> >;

  template class asynchronous_generalized_bp<canonical_gaussian<var> >;
  template class asynchronous_generalized_bp_pc<canonical_gaussian<var> >;

  template <>
  struct argument_traits<grid_vertex>
    : fixed_discrete_traits<grid_vertex, 2> { };
}

using namespace libgm;

typedef domain<grid_vertex> domain_type;
typedef probability_table<grid_vertex> ptable;

template <typename Engine>
void test(const pairwise_markov_network<ptable>& model,
          const region_graph<domain_type>& rg,
          std::size_t niters,
          const decomposable<ptable>& joint,
          double error_tol,
          double diff_tol) {
  // run the inference
  Engine engine(rg, max_diff_fn<ptable>());
  engine.initialize_factors(model);
  double residual = 0.0;
  for(std::size_t i = 0; i < niters; i++) {
    residual = engine.iterate(0.3);
  }
  BOOST_CHECK_SMALL(residual, 0.1);

  // iterate
  for (std::size_t it = 0; it < niters; ++it) {
    engine.iterate(0.5);
  }

  // check if the approximation error is small enough
  double max_error = 0;
  for (grid_vertex var : joint.arguments()) {
    ptable exact = joint.marginal({var});
    ptable approx = engine.belief({var});
    double error = max_diff(exact, approx);
    max_error = std::max(max_error, error);
    BOOST_CHECK_SMALL(error, error_tol);
  }
  std::cout << "Maximum error: " << max_error << std::endl;

  // Check if the edge marginals agree
  double max_difference = 0;
  for (auto e : rg.edges()) {
    ptable sbel = engine.belief(e.source()).marginal(rg.cluster(e.target()));
    ptable tbel = engine.belief(e.target());
    double diff = max_diff(sbel, tbel);
    max_difference = std::max(max_difference, diff);
    BOOST_CHECK_SMALL(diff, diff_tol);
  }
  std::cout << "Belief consistency: " << max_difference << std::endl;
}

struct fixture {
  fixture() {
    // generate a random model
    int m = 5;
    int n = 4;
    make_grid_graph(m, n, mn);
    uniform_table_generator<ptable> gen;
    std::mt19937 rng;
    mn.initialize(bind_marginal(gen, rng), bind_marginal(gen, rng));

    // create a region graph with clusters over pairs of adjacent variables
    std::vector<domain_type> clusters;
    for (auto e : mn.edges()) {
      clusters.push_back({e.source(), e.target()});
    }
    pairs_rg.bethe(clusters);

    // create a region graph with clusters over 2x2 adjacent variables
    std::vector<domain_type> root_clusters;
    for(int i = 0; i < m - 1; i++) {
      for(int j = 0; j < n - 1; j++) {
        domain_type cluster({{i, j}, {i+1, j}, {i, j+1}, {i+1, j+1}});
        root_clusters.push_back(cluster);
      }
    }
    square_rg.saturated(root_clusters);

    // compute the joint distribution
    dm *= mn;
  }

  pairwise_markov_network<ptable> mn;
  region_graph<domain_type> pairs_rg;
  region_graph<domain_type> square_rg;
  decomposable<ptable> dm;
};

BOOST_FIXTURE_TEST_CASE(test_parent_to_child, fixture) {
  test<asynchronous_generalized_bp_pc<ptable> >(mn, pairs_rg, 50, dm, 1e-1, 1e-3);
  test<asynchronous_generalized_bp_pc<ptable> >(mn, square_rg, 50, dm, 1e-2, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_two_way, fixture) {
  test<asynchronous_generalized_bp<ptable> >(mn, pairs_rg, 50, dm, 1e-1, 1e-3);
  test<asynchronous_generalized_bp<ptable> >(mn, square_rg, 50, dm, 1e-2, 1e-2);
}
