#define BOOST_TEST_MODULE directed_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/directed_graph.hpp>

#include <boost/mpl/list.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "../predicates.hpp"

namespace libgm {
  template class directed_graph<size_t>;
  template class directed_graph<size_t, std::string, double>;
}

using namespace libgm;

typedef directed_graph<size_t, size_t, size_t> graph_type;
typedef directed_edge<size_t> edge_type;
typedef std::pair<size_t, size_t> vpair;

BOOST_AUTO_TEST_CASE(test_directed_edge) {
  edge_type e1, e2;
  BOOST_CHECK(e1 == e2);
  BOOST_CHECK(e1.source() == e2.source());
  BOOST_CHECK(e1.source() == e1.target());
}


BOOST_AUTO_TEST_CASE(test_constructors) {
  // default constructor
  graph_type g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK(g1.vertices().empty());
  BOOST_CHECK(g1.edges().empty());

  // edge list constructor
  std::vector<vpair> vpairs = 
    {vpair(5, 2), vpair(1, 2), vpair(1, 3), vpair(1, 7), 
     vpair(2, 3), vpair(3, 4), vpair(4, 5), vpair(4, 1)};
  graph_type g2(vpairs);
  for (vpair vp : vpairs) {
    BOOST_CHECK(g2.contains(vp.first));
    BOOST_CHECK(g2.contains(vp.second));
    BOOST_CHECK(g2.contains(vp.first, vp.second));
    BOOST_CHECK(!g2.contains(vp.second, vp.first));
  }
  BOOST_CHECK(!g2.contains(8, 2));
  BOOST_CHECK(!g2.contains(8));

  // copy constructor
  graph_type g3(g2);
  for (vpair vp : vpairs) {
    BOOST_CHECK(g3.contains(vp.first));
    BOOST_CHECK(g3.contains(vp.second));
    BOOST_CHECK(g3.contains(vp.first, vp.second));
    BOOST_CHECK(!g3.contains(vp.second, vp.first));
  }
  BOOST_CHECK(!g3.contains(8, 2));
  BOOST_CHECK(!g3.contains(8));
}


BOOST_AUTO_TEST_CASE(test_vertices) {
  graph_type g;
  std::vector<size_t> verts = {1, 2, 3, 4, 5, 6, 7, 9, 10};
  std::map<size_t, size_t> vert_map;
  for (size_t v : verts) {
    vert_map[v] = v+2;
    g.add_vertex(v, v+2);
  }
  for (size_t v : g.vertices()) {
    BOOST_CHECK(vert_map.count(v));
    BOOST_CHECK_EQUAL(g[v], vert_map[v]);
    vert_map.erase(v);
  }
  BOOST_CHECK(vert_map.empty());
}


BOOST_AUTO_TEST_CASE(test_edges) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(1, 7), vpair(5, 8)};
  std::map<vpair, size_t> data;
  graph_type g;
  size_t i = 0;
  for (vpair vp : vpairs) {
    g.add_edge(vp.first, vp.second, i);
    data[vp] = i;
    ++i;
  }
  for (edge_type e : g.edges()) {
    BOOST_CHECK(data.count(vpair(e.source(), e.target())));
    BOOST_CHECK_EQUAL(data[vpair(e.source(), e.target())], g[e]);
    data.erase(vpair(e.source(), e.target()));
  }
  BOOST_CHECK(data.empty());
}


BOOST_AUTO_TEST_CASE(test_parents) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(4, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(1, 4),
     vpair(2, 6), vpair(3, 5), vpair(7, 4), vpair(5, 8)};
  graph_type g(vpairs);
  std::set<size_t> parents = {1, 3, 7};
  for (size_t v : g.parents(4)) {
    BOOST_CHECK(parents.count(v));
    parents.erase(v);
  }
  BOOST_CHECK(parents.empty());
}


BOOST_AUTO_TEST_CASE(test_children) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(4, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 2), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(4, 7), vpair(5, 8)};
  graph_type g(vpairs);
  std::set<size_t> children = {1, 2, 7, 9};
  for (size_t v : g.children(4)) {
    BOOST_CHECK(children.count(v));
    children.erase(v);
  }
  BOOST_CHECK(children.empty());
}


BOOST_AUTO_TEST_CASE(test_in_edges) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(7, 3), vpair(3, 8)};
  graph_type g;
  std::map<vpair, size_t> data;
  size_t i = 0;
  for (vpair vp : vpairs) { 
    if(vp.second == 3) {
      g.add_edge(vp.first, vp.second, i);
      data[vp] = i;
      ++i;
    }
  }
  for (edge_type e : g.in_edges(3)) { 
    vpair vp(e.source(), e.target());
    BOOST_CHECK(data.count(vp));
    BOOST_CHECK_EQUAL(g[e], data[vp]);
    data.erase(vp);
  }
  BOOST_CHECK(data.empty());
}


BOOST_AUTO_TEST_CASE(test_out_edges) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 2), vpair(4, 1),
     vpair(6, 3), vpair(3, 5), vpair(7, 3), vpair(3, 8)};
  graph_type g;
  std::map<vpair, size_t> data;
  size_t i = 0;
  for (vpair vp : vpairs) {
    if(vp.first == 3) {
      g.add_edge(vp.first, vp.second, i);
      data[vp] = i;
      ++i;
    }
  }
  for (edge_type e : g.out_edges(3)) { 
    vpair vp(e.source(), e.target());
    BOOST_CHECK(data.count(vp));
    BOOST_CHECK_EQUAL(g[e], data[vp]);
    data.erase(vp);
  }
  BOOST_CHECK(data.empty());
}


BOOST_AUTO_TEST_CASE(test_contains) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(7, 3), vpair(3, 8)};
  graph_type g(vpairs);
  for (size_t i = 1; i <= 10; ++i) {
    BOOST_CHECK(g.contains(i));
  }
  BOOST_CHECK(!g.contains(11));
  for (vpair vp : vpairs) {
    BOOST_CHECK(g.contains(vp.first, vp.second));
    BOOST_CHECK(!g.contains(vp.second, vp.first));
    BOOST_CHECK(g.contains(g.edge(vp.first, vp.second)));
  }
  BOOST_CHECK(!g.contains(0, 3));
  BOOST_CHECK(!g.contains(2, 10));
}


BOOST_AUTO_TEST_CASE(test_edge) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(1, 7), vpair(5, 8)};
  std::map<vpair, size_t> data;
  graph_type g;
  size_t i = 0; 
  for (vpair vp : vpairs) {
    g.add_edge(vp.first, vp.second, i);
    data[vp] = i; 
    ++i;
  }
  for (vpair vp : vpairs) {
    BOOST_CHECK_EQUAL(g.edge(vp.first, vp.second).source(), vp.first);
    BOOST_CHECK_EQUAL(g.edge(vp.first, vp.second).target(), vp.second);
    BOOST_CHECK_EQUAL(g[g.edge(vp.first, vp.second)], data[vp]);
  }
}


BOOST_AUTO_TEST_CASE(test_degree) {
  graph_type g;
  g.add_vertex(2, 3);
  BOOST_CHECK_EQUAL(g.in_degree(2), 0);
  BOOST_CHECK_EQUAL(g.out_degree(2), 0);
  BOOST_CHECK_EQUAL(g.degree(2), 0);

  g.add_edge(1, 2, 3);
  BOOST_CHECK_EQUAL(g.in_degree(2), 1);
  BOOST_CHECK_EQUAL(g.in_degree(1), 0);
  BOOST_CHECK_EQUAL(g.out_degree(1), 1);
  BOOST_CHECK_EQUAL(g.out_degree(2), 0);
  BOOST_CHECK_EQUAL(g.degree(1), 1);
  BOOST_CHECK_EQUAL(g.degree(2), 1);

  std::vector<vpair> vpairs =
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(7, 3), vpair(3, 8)};
  graph_type g2(vpairs);
  BOOST_CHECK_EQUAL(g2.in_degree(3), 3);
  BOOST_CHECK_EQUAL(g2.out_degree(4), 2);
  BOOST_CHECK_EQUAL(g2.in_degree(3) + g2.out_degree(3), g2.degree(3));
}


BOOST_AUTO_TEST_CASE(test_num) {
  std::vector<vpair> vpairs = 
    {vpair(9, 2), vpair(1, 9), vpair(1, 3), vpair(1, 10), 
     vpair(2, 3), vpair(3, 4), vpair(4, 9), vpair(4, 1),
     vpair(2, 6), vpair(3, 5), vpair(7, 3), vpair(3, 8)};
  graph_type g(vpairs);
  BOOST_CHECK_EQUAL(g.num_vertices(), 10);
  BOOST_CHECK_EQUAL(g.num_edges(), 12);
  g.remove_in_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 10);
  BOOST_CHECK_EQUAL(g.num_edges(), 9);
  g.remove_out_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 10);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);

  g.remove_vertex(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 9);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);
  g.remove_vertex(9);
  BOOST_CHECK_EQUAL(g.num_vertices(), 8);
  BOOST_CHECK_EQUAL(g.num_edges(), 3);
}


typedef boost::mpl::list<int, double, void_> test_types;
BOOST_AUTO_TEST_CASE_TEMPLATE(test_comparison, EP, test_types) {
  typedef directed_graph<size_t, int, EP> Graph;

  Graph g1;
  g1.add_vertex(1, 0);
  g1.add_vertex(2, 1);
  g1.add_vertex(3, 2);
  g1.add_edge(1, 2);
  g1.add_edge(2, 3);
  
  Graph g2;
  g2.add_vertex(1, 0);
  g2.add_vertex(3, 2);
  g2.add_vertex(2, 1);
  g2.add_edge(1, 2);
  g2.add_edge(2, 3);

  BOOST_CHECK_EQUAL(g1, g2);

  Graph g3 = g2;
  g3[1] = -1;
  BOOST_CHECK_NE(g2, g3);

  Graph g4 = g2;
  g4.remove_edge(2, 3);
  g4.add_edge(1, 3);
  BOOST_CHECK_NE(g2, g3);
}


BOOST_AUTO_TEST_CASE(test_serialization) {
  directed_graph<int, std::string, double> g;
  g.add_vertex(1, "hello");
  g.add_vertex(2, "bye");
  g.add_vertex(3, "maybe");
  g.add_edge(1, 2, 1.5);
  g.add_edge(2, 3, 2.5);
  g.add_edge(3, 2, 3.5);
  BOOST_CHECK(serialize_deserialize(g));
}


// the following test is too slow for a unit test;
// it should be in timings instead
// BOOST_AUTO_TEST_CASE(test_large) {
//   size_t n = 100000;
//   size_t m = 100;  
//   directed_graph<size_t, size_t, size_t> g;
//   for(size_t u = 0; u < n; u++) {
//     g.add_vertex(u,u);
//     for(size_t v = 0; v < m; v++) {
//       g.add_vertex(v,v);
//       g.add_edge(u,v, u * v);
//     } 
//   }
// }
