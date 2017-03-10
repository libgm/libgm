#define BOOST_TEST_MODULE undirected_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/grid_graph.hpp>

#include <boost/mpl/list.hpp>

#include <unordered_map>
#include <vector>

#include "../predicates.hpp"

namespace libgm {
  template class grid_graph<>;
  template class grid_graph<int, std::string, double>;
}

using namespace libgm;

using graph_type = grid_graph<int, std::size_t, std::size_t>;

BOOST_AUTO_TEST_CASE(test_grid_edge) {
  grid_vertex<int> v(1, 2);
  grid_edge<int> e1(v, v.right()), e2(v.left(), v);
  BOOST_CHECK_EQUAL(e1, e1);
  BOOST_CHECK_EQUAL(e1, e2.reversed());
  BOOST_CHECK_NE(e1, e2);
}


BOOST_AUTO_TEST_CASE(test_constructors) {
  // default constructor
  graph_type g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK(g1.vertices().empty());
  BOOST_CHECK(g1.edges().empty());

  // size constructor
  graph_type g2(3, 4);
  BOOST_CHECK(!g2.empty());
  BOOST_CHECK_EQUAL(g2.rows(), 3);
  BOOST_CHECK_EQUAL(g2.cols(), 4);
  BOOST_CHECK_EQUAL(g2.num_vertices(), 12);
  BOOST_CHECK_EQUAL(g2.num_edges(), 2*4 + 3*3);

  // copy constructor
  grid_vertex v(1, 2);
  g2[v] = 4;
  g2(v, v.right()) = 5;
  graph_type g3(g2);
  BOOST_CHECK(!g3.empty());
  BOOST_CHECK_EQUAL(g3.rows(), 3);
  BOOST_CHECK_EQUAL(g3.cols(), 4);
  BOOST_CHECK_EQUAL(g3.num_vertices(), 12);
  BOOST_CHECK_EQUAL(g3.num_edges(), 2*4 + 3*3);
  BOOST_CHECK_EQUAL(g2, g3);
}


BOOST_AUTO_TEST_CASE(test_vertices) {
  graph_type g(3, 4);
  graph_type::vertex_iterator it, end;
  std::tie(it, end) = g.vertices();
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 3; ++row) {
      BOOST_CHECK_NE(it, end);
      BOOST_CHECK_EQUAL(it->row, row);
      BOOST_CHECK_EQUAL(it->col, col);
      ++it;
    }
  }
  BOOST_CHECK_EQUAL(it, end);
}


BOOST_AUTO_TEST_CASE(test_edges) {
  graph_type g(3, 4);
  graph_type::edge_iterator it, end;
  std::tie(it, end) = g.edges();
  grid_vertex<int> v;
  for (int v.col = 0; v.col < 4; ++v.col) {
    for (int v.row = 0; v.row < 3; ++v.row) {
      if (v.row < 2) { // vertical edge
        BOOST_CHECK_NE(it, end);
        BOOST_CHECK_EQUAL(it->source(), v);
        BOOST_CHECK_EQUAL(it->target(), v.below());
        ++it;
      }
      if (v.col < 3) { // horizontal edge
        BOOST_CHECK_NE(it, end);
        BOOST_CHECK_EQUAL(it->source(), v);
        BOOST_CHECK_EQUAL(it->target(), v.right());
        ++it;
      }
    }
  }
  BOOST_CHECK_EQUAL(it, end);
}


struct neighbor_fixture {
  neighbor_fixture()
    : g(3, 4) {
    grid_vertex<int> u(0, 0);
    grid_vertex<int> v(2, 0);
    grid_vertex<int> w(0, 3);
    grid_vertex<int> q(2, 3);
    grid_vertex<int> r(1, 2);
    nbrs[u] = { u.below(), u.right() };
    nbrs[v] = { v.above(), u.right() };
    nbrs[w] = { w.left(), w.below() };
    nbrs[q] = { q.above(), q.left() };
    nbrs[r] = { r.above(), r.left(), r.below(), r.right() };
  }

  std::unordered_map<grid_vertex<int>, std::vector<grid_vertex<int> > > nbr_map;
  graph_type g;
};


BOOST_FIXTURE_TEST_CASE(test_neighbors, neighbor_fixture) {
  for (const auto& nbrs : nbr_map) {
    std::size_t i = 0;
    for (grid_vertex<int> v : g.neighbors(nbrs.first)) {
      BOOST_CHECK_EQUAL(v, nbrs.second[i++]);
    }
    BOOST_CHECK_EQUAL(nbrs.second.size(), i);
  }
}


BOOST_FIXTURE_TEST_CASE(test_in_edges, neighbor_fixture) {
  for (const auto& nbrs : nbr_map) {
    std::size_t i = 0;
    for (grid_edge<int> e : g.in_edges(nbrs.first)) {
      BOOST_CHECK_EQUAL(e, make_grid_edge(nbrs.second[i++], nbrs.first));
    }
    BOOST_CHECK_EQUAL(nbrs.second.size(), i);
  }
}


BOOST_FIXTURE_TEST_CASE(test_out_edges, neighbor_fixture) {
  for (const auto& nbrs : nbr_map) {
    std::size_t i = 0;
    for (grid_edge<int> e : g.out_edges(nbrs.first)) {
      BOOST_CHECK_EQUAL(e, make_grid_edge(nbrs.first, nbrs.second[i++]));
    }
    BOOST_CHECK_EQUAL(nbrs.second.size(), i);
  }
}


BOOST_FIXTURE_TEST_CASE(test_degree, neighbor_fixture) {
  for (const auto& nbrs : nbr_map) {
    BOOST_CHECK(g.degree(nbrs.first), nbrs.second.size());
  }
}


BOOST_AUTO_TEST_CASE(test_contains) {
  graph_type g(3, 4);
  BOOST_CHECK(g.contains(grid_vertex<int>(2, 3)));
  BOOST_CHECK(g.contains(grid_vertex<int>(0, 0)));
  BOOST_CHECK(g.contains(grid_vertex<int>(0, 1)));
  BOOST_CHECK(!g.contains(grid_vertex<int>(-1, 0)));
  BOOST_CHECK(!g.contains(grid_vertex<int>(3, 3)));
  BOOST_CHECK(g.contains(grid_vertex<int>(0, 1), grid_vertex<int>(0, 2)));
  BOOST_CHECK(g.contains(grid_vertex<int>(0, 1), grid_vertex<int>(0, 0)));
  BOOST_CHECK(g.contains(grid_vertex<int>(0, 1), grid_vertex<int>(1, 1)));
  BOOST_CHECK(!g.contains(grid_vertex<int>(0, 1), grid_vertex<int>(1, 2)));
  BOOST_CHECK(!g.contians(grid_vertex<int>(0, 1), grid_vertex<int>(2, 0)));
}


BOOST_AUTO_TEST_CASE(test_edge) {
  graph_type g(3, 4);
  grid_vertex<int> u(0, 1);
  grid_vertex<int> v(0, 2);
  grid_vertex<int> w(1, 1);
  BOOST_CHECK(g.edge(u, v), make_grid_edge(u, v));
  BOOST_CHECK(g.edge(u, w), make_grid_edge(u, w));
}


BOOST_AUTO_TEST_CASE(test_serialization) {
  grid_graph<int, std::string, double> g(3, 4);
  g(1, 2) = "hello";
  g(0, 1) = "world";
  g[grid_edge<int>(0, 0, 1, 0)] = 1.5;
  g[grid_edge<int>(2, 2, 1, 2)] = 2.5;
  BOOST_CHECK(serialize_deserialize(g));
}
