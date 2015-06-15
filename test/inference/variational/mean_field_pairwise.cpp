#define BOOST_TEST_MODULE mean_field_pairwise
#include <boost/test/unit_test.hpp>

#include <libgm/inference/variational/mean_field_pairwise.hpp>

#include <libgm/factor/canonical_array.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/bind.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <random>

namespace libgm {
  template <>
  struct argument_traits<std::pair<int, int> >
    : fixed_discrete_traits<std::pair<int, int>, 2> { };
}

using namespace libgm;

typedef std::pair<int, int> grid_vertex;
typedef canonical_array<double, 1, grid_vertex> carray1_type;
typedef canonical_array<double, 2, grid_vertex> carray2_type;
typedef probability_array<double, 1, grid_vertex> parray1_type;
typedef probability_table<double, grid_vertex> ptable_type;

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace libgm;

  std::size_t m = 8;
  std::size_t n = 5;
  std::size_t niters = 20;

  // Create a random grid Markov network
  std::mt19937 rng;
  pairwise_markov_network<carray1_type, carray2_type> model;
  make_grid_graph(m, n, model);
  model.initialize(bind_marginal(uniform_table_generator<carray1_type>(), rng),
                   bind_marginal(uniform_table_generator<carray2_type>(), rng));

  // run exact inference
  pairwise_markov_network<ptable_type> converted(model);
  sum_product_calibrate<ptable_type> sp(converted);
  std::cout << "Tree width of the model: " << sp.jt().tree_width() << std::endl;
  sp.calibrate();
  sp.normalize();
  std::cout << "Finished exact inference" << std::endl;

  // run mean field inference
  mean_field_pairwise<carray1_type, carray2_type> mf(&model);
  double diff;
  for (std::size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
    std::cout << "Iteration " << it << ": " << diff << std::endl;
  }
  BOOST_CHECK_LT(diff, 1e-4);

  // compute the KL divergence from the exact to mean field result
  double kl = 0.0;
  for (grid_vertex v : model.vertices()) {
    kl += kl_divergence(parray1_type(sp.belief({v})), mf.belief(v));
  }
  kl /= model.num_vertices();
  std::cout << "Average kl = " << kl << std::endl;
  BOOST_CHECK_LT(kl, 0.02);
}
