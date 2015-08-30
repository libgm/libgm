#define BOOST_TEST_MODULE pairwise_mn_bp
#include <boost/test/unit_test.hpp>

#include <libgm/inference/loopy/pairwise_mn_bp.hpp>

#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

typedef std::pair<int, int> grid_vertex;

namespace libgm {
  template class synchronous_pairwise_mn_bp<probability_table<var> >;
  template class asynchronous_pairwise_mn_bp<probability_table<var> >;
  template class residual_pairwise_mn_bp<probability_table<var> >;
  template class exponential_pairwise_mn_bp<probability_table<var> >;

  template <>
  struct argument_traits<grid_vertex>
    : fixed_continuous_traits<grid_vertex, 1> { };
}

using namespace libgm;

typedef moment_gaussian<grid_vertex> mgaussian;
typedef canonical_gaussian<grid_vertex> cgaussian;

void test(pairwise_mn_bp<cgaussian>&& engine,
          std::size_t niters,
          const mgaussian& joint,
          double error) {
  for (std::size_t i = 0; i < niters; ++i) {
    engine.iterate(1.0);
  }

  // check that the marginal means have converged to the true means
  for (grid_vertex v : joint.arguments()) {
    mgaussian belief(engine.belief(v));
    BOOST_CHECK_SMALL(belief.mean(v) - joint.mean(v), error);
  }

  // check that the edge marginals agree on the shared variable
  for (grid_vertex v : engine.graph().vertices()) {
    cgaussian nbelief = engine.belief(v);
    for (auto e : engine.graph().in_edges(v)) {
      cgaussian ebelief = engine.belief(e);
      BOOST_CHECK_SMALL(max_diff(nbelief, ebelief.marginal({v})), error);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  std::size_t m = 5;
  std::size_t n = 4;

  // construct a grid network with attractive Gaussian potentials
  pairwise_markov_network<cgaussian> model;
  make_grid_graph(m, n, model);
  moment_gaussian_generator<grid_vertex> gen;
  std::mt19937 rng;
  model.initialize(nullptr, bind_marginal<cgaussian>(gen, rng));

  // run exact inference and the various loopy BP algorithms & compare results
  mgaussian joint(prod_all(model));
  diff_fn<cgaussian> diff = max_diff_fn<cgaussian>();
  test(synchronous_pairwise_mn_bp<cgaussian>(&model, diff), 10, joint, 1e-5);
  test(asynchronous_pairwise_mn_bp<cgaussian>(&model, diff), 10, joint, 1e-5);
  test(residual_pairwise_mn_bp<cgaussian>(&model, diff), m*n*10, joint, 1e-5);
}
