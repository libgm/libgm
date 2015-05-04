#define BOOST_TEST_MODULE tree_traversal
#include <boost/test/unit_test.hpp>

#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/undirected_graph.hpp>

using namespace libgm;

struct fixture {
  fixture() {
    g.add_edge(5, 2);
    g.add_edge(2, 1);
    g.add_edge(1, 7);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
  }
  typedef undirected_graph<int> graph_type;
  typedef undirected_edge<int> edge_type;
  typedef std::pair<int, int> vpair;
  graph_type g;
};

BOOST_FIXTURE_TEST_CASE(test_preorder, fixture) {
  std::set<int> visited({1});
  pre_order_traversal(g, 1, [&](const edge_type& e) {
      BOOST_CHECK(visited.count(e.source()));
      visited.insert(e.target());
    });
}

BOOST_FIXTURE_TEST_CASE(test_postorder, fixture) {
  std::set<int> visited;
  for (int v : g.vertices()) {
    if (g.degree(v) == 1) {
      visited.insert(v);
    }
  }
  post_order_traversal(g, 1, [&](const edge_type& e) {
      BOOST_CHECK(visited.count(e.source()));
      visited.insert(e.target());
    });
}


BOOST_FIXTURE_TEST_CASE(test_mpp, fixture) {
  std::set<vpair> visited;
  mpp_traversal(g, 0, [&](const edge_type& e) {
      for (edge_type in : g.in_edges(e.source())) {
        if (in != e) {
          BOOST_CHECK(visited.count(in.pair()));
        }
      }
      visited.insert(e.pair());
    });
}
