#define BOOST_TEST_MODULE pairwise_mn_bp
#include <boost/test/unit_test.hpp>

#include <libgm/inference/loopy/pairwise_mn_bp.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/functional.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>
#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

namespace libgm {
  template class synchronous_pairwise_mn_bp<cgaussian>;
  template class asynchronous_pairwise_mn_bp<cgaussian>;
  template class residual_pairwise_mn_bp<cgaussian>;
  template class exponential_pairwise_mn_bp<cgaussian>;

  template class synchronous_pairwise_mn_bp<ptable>;
  template class asynchronous_pairwise_mn_bp<ptable>;
  template class residual_pairwise_mn_bp<ptable>;
  template class exponential_pairwise_mn_bp<ptable>;
}

using namespace libgm;

void test(pairwise_mn_bp<cgaussian>&& engine,
          std::size_t niters,
          const mgaussian& joint,
          double error) {
  for (std::size_t i = 0; i < niters; ++i) {
    engine.iterate(1.0);
  }

  // check that the marginal means have converged to the true means
  for (variable v : joint.arguments()) {
    mgaussian belief(engine.belief(v));
    BOOST_CHECK_SMALL((belief.mean(v) - joint.mean(v)).norm(), error);
  }

  // check that the edge marginals agree on the shared variable
  for (variable v : engine.graph().vertices()) {
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
  universe u;
  domain variables = u.new_continuous_variables(m * n, "v", 1);
  pairwise_markov_network<cgaussian> model;
  make_grid_graph(variables, m, n, model);
  moment_gaussian_generator<double> gen;
  std::mt19937 rng;
  model.initialize(nullptr, marginal_fn<cgaussian>(gen, rng));

  // run exact inference and the various loopy BP algorithms & compare results
  mgaussian joint(prod_all(model));
  diff_fn<cgaussian> diff = max_diff_fn<cgaussian>();
  test(synchronous_pairwise_mn_bp<cgaussian>(&model, diff), 10, joint, 1e-5);
  test(asynchronous_pairwise_mn_bp<cgaussian>(&model, diff), 10, joint, 1e-5);
  test(residual_pairwise_mn_bp<cgaussian>(&model, diff), m*n*10, joint, 1e-5);
}
