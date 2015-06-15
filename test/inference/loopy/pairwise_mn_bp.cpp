#define BOOST_TEST_MODULE pairwise_mn_bp
#include <boost/test/unit_test.hpp>

#include <libgm/inference/loopy/pairwise_mn_bp.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

namespace libgm {
  template class synchronous_pairwise_mn_bp<ptable>;
  template class asynchronous_pairwise_mn_bp<ptable>;
  template class residual_pairwise_mn_bp<ptable>;
  template class exponential_pairwise_mn_bp<ptable>;

  template <>
  struct argument_traits<std::pair<int, int> >
    : fixed_continuous_traits<std::pair<int, int>, 1> { };
}

using namespace libgm;

typedef std::pair<int, int> grid_vertex;
typedef moment_gaussian<double, grid_vertex> mgaussian_type;
typedef canonical_gaussian<double, grid_vertex> cgaussian_type;

void test(pairwise_mn_bp<cgaussian_type>&& engine,
          std::size_t niters,
          const mgaussian_type& joint,
          double error) {
  for (std::size_t i = 0; i < niters; ++i) {
    engine.iterate(1.0);
  }

  // check that the marginal means have converged to the true means
  for (grid_vertex v : joint.arguments()) {
    mgaussian_type belief(engine.belief(v));
    BOOST_CHECK_SMALL((belief.mean(v) - joint.mean(v)).norm(), error);
  }

  // check that the edge marginals agree on the shared variable
  for (grid_vertex v : engine.graph().vertices()) {
    cgaussian_type nbelief = engine.belief(v);
    for (auto e : engine.graph().in_edges(v)) {
      cgaussian_type ebelief = engine.belief(e);
      BOOST_CHECK_SMALL(max_diff(nbelief, ebelief.marginal({v})), error);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  std::size_t m = 5;
  std::size_t n = 4;

  // construct a grid network with attractive Gaussian potentials
  pairwise_markov_network<cgaussian_type> model;
  make_grid_graph(m, n, model);
  moment_gaussian_generator<double, grid_vertex> gen;
  std::mt19937 rng;
  model.initialize(nullptr, bind_marginal<cgaussian_type>(gen, rng));

  // run exact inference and the various loopy BP algorithms & compare results
  mgaussian_type joint(prod_all(model));
  diff_fn<cgaussian_type> diff = max_diff_fn<cgaussian_type>();
  test(synchronous_pairwise_mn_bp<cgaussian_type>(&model, diff), 10, joint, 1e-5);
  test(asynchronous_pairwise_mn_bp<cgaussian_type>(&model, diff), 10, joint, 1e-5);
  test(residual_pairwise_mn_bp<cgaussian_type>(&model, diff), m*n*10, joint, 1e-5);
}
