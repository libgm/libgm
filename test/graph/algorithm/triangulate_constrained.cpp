#define BOOST_TEST_MODULE constrained_triangulation
#include <boost/test/unit_test.hpp>

#include <libgm/argument/basic_domain.hpp>
#include <libgm/graph/algorithm/constrained_elim_strategy.hpp>
#include <libgm/graph/algorithm/min_degree_strategy.hpp>
#include <libgm/graph/algorithm/min_fill_strategy.hpp>
#include <libgm/graph/algorithm/triangulate.hpp>
#include <libgm/graph/special/grid_graph.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/graph/undirected_graph.hpp>

namespace libgm {
  template <>
  struct argument_traits<std::pair<int, int>>
    : fixed_discrete_traits<std::pair<int, int>, 2> { };
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_triangulation) {
  typedef std::pair<int, int> grid_vertex;
  typedef undirected_graph<grid_vertex> graph_type;
  typedef basic_domain<grid_vertex> domain_type;

  // Build a 2 x 5 lattice
  graph_type lattice;
  make_grid_graph(2, 5, lattice);

  /*
    0,0 - 0,1 - 0,2 - 0,3 - 0,4
     |     |     |     |     |
    1,0 - 1,1 - 1,2 - 1,3 - 1,4
  */

  // Create a constrained elimination strategy that eliminates row 1 first
  auto s = constrained_strategy<int>([](grid_vertex v) { return v.first; },
                                     min_degree_strategy());

  // Create a junction tree using this elimination strategy.
  // If we imagine vertices in row 0 as being discrete and vertices in row 1
  // as continuous random variables, then this creates a strongly-rooted
  // junction tree.
  cluster_graph<domain_type> jt;
  triangulate_maximal<domain_type>(lattice, [&](domain_type&& clique) {
      jt.add_cluster(clique.sort());
    }, s);
  jt.mst_edges();

  std::vector<domain_type> cliques = {
    {{0, 4}, {1, 3}, {1, 4}},
    {{0, 0}, {1, 0}, {1, 1}},
    {{0, 0}, {0, 1}, {1, 1}, {1, 2}},
    {{0, 3}, {0, 4}, {1, 2}, {1, 3}},
    {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}}
  };
  cluster_graph<domain_type> jt2;
  jt2.triangulated(cliques);

  BOOST_CHECK_EQUAL(jt, jt2);
}
