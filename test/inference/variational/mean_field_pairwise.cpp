#define BOOST_TEST_MODULE mean_field_pairwise
#include <boost/test/unit_test.hpp>

#include <libgm/inference/variational/mean_field_pairwise.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_array.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/functional.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace libgm;

  std::size_t m = 8;
  std::size_t n = 5;
  std::size_t niters = 20;

  // Create a random grid Markov network
  universe u;
  std::mt19937 rng;
  pairwise_markov_network<carray1, carray2> model;
  domain vars = u.new_finite_variables(m * n, "v", 2);
  make_grid_graph(vars, m, n, model);
  model.initialize(marginal_fn(uniform_table_generator<carray1>(), rng),
                   marginal_fn(uniform_table_generator<carray2>(), rng));

  // run exact inference
  pairwise_markov_network<ptable> converted(model);
  sum_product_calibrate<ptable> sp(converted);
  std::cout << "Tree width of the model: " << sp.jt().tree_width() << std::endl;
  sp.calibrate();
  sp.normalize();
  std::cout << "Finished exact inference" << std::endl;

  // run mean field inference
  mean_field_pairwise<carray1, carray2> mf(&model);
  double diff;
  for (std::size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
    std::cout << "Iteration " << it << ": " << diff << std::endl;
  }
  BOOST_CHECK_LT(diff, 1e-4);

  // compute the KL divergence from the exact to mean field result
  double kl = 0.0;
  for (variable v : model.vertices()) {
    kl += kl_divergence(parray1(sp.belief({v})), mf.belief(v));
  }
  kl /= model.num_vertices();
  std::cout << "Average kl = " << kl << std::endl;
  BOOST_CHECK_LT(kl, 0.02);
}
