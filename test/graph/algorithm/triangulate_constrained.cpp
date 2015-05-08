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


struct elim_priority_functor {
  typedef std::size_t result_type;
  template <typename Graph>
  std::size_t operator()(typename Graph::vertex_type v, const Graph& graph) {
    return graph[v];
  }
};

namespace libgm {
  template class constrained_elim_strategy<
    ::elim_priority_functor, min_degree_strategy>;
  template class constrained_elim_strategy<
    ::elim_priority_functor, min_fill_strategy>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_triangulation) {
  typedef undirected_graph<std::size_t, std::size_t> graph_type;
  typedef basic_domain<std::size_t> domain_type;

  // Build an m x 2 lattice.
  std::size_t m = 5;
  graph_type lattice;

  // Add the vertices and prioritize their elimination so that
  // vertices in the second column (ids 6-10) have a lower
  // elimination priority than vertices in the first column (ids 1-5).
  arma::umat v = make_grid_graph(m, 2, lattice);
  for(std::size_t i = 0; i < m; i++) {
    for(std::size_t j = 0; j < 2; j++) {
      lattice[v(i,j)] = j;
    }
  }

  // Create a constrained elimination strategy (using min-degree as
  // the secondary strategy).
  constrained_elim_strategy<elim_priority_functor, min_degree_strategy> s;

  // Create a junction tree using this elimination strategy.
  // (If we imagine vertices 1-5 as being discrete and vertices 6-10 as
  // continuous random variables, then this creates a strongly-rooted
  // junction tree.)
  cluster_graph<domain_type> jt;
  triangulate_maximal<domain_type>(lattice, [&](domain_type&& clique) {
      jt.add_cluster(clique.sort());
    }, s);
  jt.mst_edges();

  // Vertices
  // 1: ({5 9 10}  0)
  // 2: ({1 6 7}  0)
  // 3: ({1 2 7 8}  0)
  // 4: ({4 5 8 9}  0)
  // 5: ({1 2 3 4 5 8}  0)

  // Edges
  // 4 -- 5
  // 3 -- 5
  // 2 -- 3
  // 1 -- 4

  std::vector<domain_type> cliques = {
    {5, 9, 10},
    {1, 6, 7},
    {1, 2, 7, 8},
    {4, 5, 8, 9},
    {1, 2, 3, 4, 5, 8}};
  cluster_graph<domain_type> jt2;
  jt2.triangulated(cliques);

  BOOST_CHECK_EQUAL(jt, jt2);
}
