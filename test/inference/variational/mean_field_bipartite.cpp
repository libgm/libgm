#define BOOST_TEST_MODULE mean_field_bipartite
#include <boost/test/unit_test.hpp>

#include <libgm/inference/variational/mean_field_bipartite.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_array.hpp>
#include <libgm/factor/probability_array.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/graph/bipartite_graph.hpp>
#include <libgm/inference/exact/sum_product_calibrate.hpp>

#include <boost/serialization/strong_typedef.hpp>

#include <random>

using namespace libgm;

typedef variable vertex_type;
BOOST_STRONG_TYPEDEF(vertex_type, vertex1);
BOOST_STRONG_TYPEDEF(vertex_type, vertex2);

namespace std {
  template<>
  struct hash<::vertex1> {
    typedef ::vertex1 argument_type;
    typedef std::size_t result_type;
    std::size_t operator()(::vertex1 v) const { return hash_value(v.t); }
  };

  template<>
  struct hash<::vertex2> {
    typedef ::vertex2 argument_type;
    typedef std::size_t result_type;
    std::size_t operator()(::vertex2 v) const { return hash_value(v.t); }
  };
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  std::size_t nvertices = 20;
  std::size_t nedges = 50;
  std::size_t niters = 20;

  // Create a random bipartite graph
  universe u;
  bipartite_graph<vertex1, vertex2, carray1, carray1, carray2> model;
  std::vector<vertex1> v1;
  uniform_table_generator<carray1> node_gen;
  uniform_table_generator<carray2> edge_gen;
  std::mt19937 rng;
  std::vector<ptable> factors;

  // node potentials
  for (std::size_t i = 0; i < nvertices; ++i) {
    vertex1 v1(u.new_discrete_variable("x" + std::to_string(i), 2));
    vertex2 v2(u.new_discrete_variable("y" + std::to_string(i), 2));
    model.add_vertex(v1, node_gen({v1.t}, rng));
    model.add_vertex(v2, node_gen({v2.t}, rng));
    factors.emplace_back(model[v1]);
    factors.emplace_back(model[v2]);
  }

  // edge potentials
  for (std::size_t i = 0; i < nedges; /* advanced on success */) {
    vertex1 v1 = model.sample_vertex1(rng);
    vertex2 v2 = model.sample_vertex2(rng);
    auto result = model.add_edge(v1, v2, edge_gen({v1.t, v2.t}, rng));
    if (result.second) {
      factors.emplace_back(model[result.first]);
      ++i;
    }
  }

  // run exact inference
  sum_product_calibrate<ptable> sp(factors);
  std::cout << "Tree width of the model: " << sp.jt().tree_width() << std::endl;
  sp.calibrate();
  sp.normalize();
  std::cout << "Finished exact inference" << std::endl;

  // run mean field inference
  mean_field_bipartite<vertex1, vertex2, carray1, carray2> mf(&model, 4);
  double diff;
  for (std::size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
    std::cout << "Iteration " << it << ": " << diff << std::endl;
  }
  BOOST_CHECK_LT(diff, 1e-4);

  // compute the KL divergence from exact to mean field
  double kl1 = 0.0;
  double kl2 = 0.0;
  for (vertex1 v : model.vertices1()) {
    kl1 += kl_divergence(parray1(sp.belief({v.t})), mf.belief(v));
  }
  for (vertex2 v : model.vertices2()) {
    kl2 += kl_divergence(parray1(sp.belief({v.t})), mf.belief(v));
  }
  kl1 /= nvertices;
  kl2 /= nvertices;
  std::cout << "Average kl1 = " << kl1 << std::endl;
  std::cout << "Average kl2 = " << kl2 << std::endl;
  BOOST_CHECK_LT(kl1, 0.02);
  BOOST_CHECK_LT(kl2, 0.02);
}
