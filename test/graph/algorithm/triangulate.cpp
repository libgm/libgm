#define BOOST_TEST_MODULE triangulate
#include <boost/test/unit_test.hpp>

#include <libgm/graph/algorithm/min_fill_strategy.hpp>
#include <libgm/graph/algorithm/min_degree_strategy.hpp>
#include <libgm/graph/algorithm/triangulate.hpp>
#include <libgm/graph/undirected_graph.hpp>

#include <boost/mpl/list.hpp>

#include <random>
#include <set>

using namespace libgm;

typedef boost::mpl::list<min_fill_strategy, min_degree_strategy> strategies;
BOOST_AUTO_TEST_CASE_TEMPLATE(test_coverage, Strategy, strategies) {
  std::size_t nvertices = 20;
  std::size_t nedges = 80;
  typedef std::vector<std::size_t> clique_type;

  // create a random graph with given number of vertices and edges
  undirected_graph<std::size_t> g;
  std::mt19937 rng;
  std::uniform_int_distribution<std::size_t> unif(1, nvertices);
  std::set<std::pair<std::size_t, std::size_t> > edges;
  for (std::size_t i = 0; i < nvertices; ++i) {
    g.add_vertex(i+1);
  }
  for (std::size_t i = 0; i < nedges; ++i) {
    edges.insert(g.add_edge(unif(rng), unif(rng)).first.unordered_pair());
  }

  // make a backup (triangulation is a mutating operation)
  undirected_graph<std::size_t> g2 = g;
  std::set<std::pair<std::size_t, std::size_t> > edges2 = edges;

  // triangulate and check if the cliques cover all the edges
  triangulate<clique_type>(g, [&](const clique_type& clique) {
      for (std::size_t i = 0; i < clique.size(); ++i) {
        for (std::size_t j = i; j < clique.size(); ++j) {
          edges.erase(std::minmax(clique[i], clique[j]));
        }
      }
    }, Strategy());
  BOOST_CHECK(edges.empty());

  // compute maximal cliques and check if they cover all the edges
  triangulate_maximal<clique_type>(g2, [&](const clique_type& clique) {
      for (std::size_t i = 0; i < clique.size(); ++i) {
        for (std::size_t j = i; j < clique.size(); ++j) {
          edges2.erase(std::minmax(clique[i], clique[j]));
        }
      }
    }, Strategy());
  BOOST_CHECK(edges2.empty());
}
