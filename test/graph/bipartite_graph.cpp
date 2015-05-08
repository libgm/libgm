#define BOOST_TEST_MODULE bipartite_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/bipartite_graph.hpp>

#include <boost/mpl/list.hpp>
#include <boost/range/algorithm.hpp>

#include <map>
#include <set>
#include <vector>

#include "../predicates.hpp"

namespace libgm {
  template class bipartite_graph<std::string, std::size_t>;
  template class bipartite_graph<std::string, std::size_t,
                                 double, double, double>;
}

using namespace libgm;
using std::string;

typedef bipartite_graph<string, std::size_t, std::size_t, std::size_t,
                        std::size_t> graph_type;
// using strings for keys is hugely inefficient and generally not recommended,
// but it's ok for short unit tests
// in addition, using std::string is good to ensure that operator bool() is
// not required for vertices by our implementation

typedef graph_type::edge_type edge_type;
typedef std::pair<string, std::size_t> vpair;

BOOST_TEST_DONT_PRINT_LOG_VALUE(graph_type::vertex1_iterator);
BOOST_TEST_DONT_PRINT_LOG_VALUE(graph_type::vertex2_iterator);
BOOST_TEST_DONT_PRINT_LOG_VALUE(graph_type::edge_iterator);

BOOST_AUTO_TEST_CASE(test_constructors) {
  // default constructor
  graph_type ga;
  BOOST_CHECK(ga.empty());
  BOOST_CHECK_EQUAL(ga.edges().begin(), ga.edges().end());
  BOOST_CHECK_EQUAL(ga.vertices1().begin(), ga.vertices1().end());
  BOOST_CHECK_EQUAL(ga.vertices2().begin(), ga.vertices2().end());

  // edge list constructor
  std::vector<vpair> vertex_pairs =
    {vpair("0", 2), vpair("1", 2), vpair("1", 3), vpair("1", 7),
     vpair("2", 3), vpair("3", 4), vpair("4", 3), vpair("4", 1)};
  graph_type gb(vertex_pairs);
  for (vpair vp : vertex_pairs) {
    BOOST_CHECK(gb.contains(vp.first));
    BOOST_CHECK(gb.contains(vp.second));
    BOOST_CHECK(gb.contains(vp.first, vp.second));
  }
  BOOST_CHECK(!gb.contains("z"));
  BOOST_CHECK(!gb.contains(8));
  BOOST_CHECK(!gb.contains("8", 2));
  BOOST_CHECK_EQUAL(gb.num_edges(), 8);

  // copy constructor
  graph_type gc(gb);
  for (vpair vp : vertex_pairs) {
    BOOST_CHECK(gc.contains(vp.first));
    BOOST_CHECK(gc.contains(vp.second));
    BOOST_CHECK(gc.contains(vp.first, vp.second));
  }
  BOOST_CHECK(!gc.contains("8", 2));
  BOOST_CHECK(!gc.contains(8));
  BOOST_CHECK(!gc.contains("z"));
  BOOST_CHECK_EQUAL(gc.num_edges(), 8);
}


BOOST_AUTO_TEST_CASE(test_vertices) {
  graph_type g;
  std::vector<int> verts = {{1,2,3,4,5,6,7,8,9,10}};
  std::map<string, std::size_t> vert1_map;
  std::map<std::size_t, std::size_t> vert2_map;
  for (int v : verts) {
    std::string u(1, v + 'A');
    vert1_map[u] = v;
    vert2_map[v] = v;
    g.add_vertex(u, v);
    g.add_vertex(v, v);
  }
  for (string v : g.vertices1()) {
    BOOST_CHECK(vert1_map.count(v) == 1);
    BOOST_CHECK_EQUAL(g[v], vert1_map[v]);
    vert1_map.erase(v);
  }
  for (std::size_t v : g.vertices2()) {
    BOOST_CHECK(vert2_map.count(v) == 1);
    BOOST_CHECK_EQUAL(g[v], vert2_map[v]);
    vert2_map.erase(v);
  }
  BOOST_CHECK(vert1_map.empty());
  BOOST_CHECK(vert2_map.empty());
}

struct fixture {
  fixture() {
    std::vector<vpair> connected_pairs =
      {vpair("0", 2), vpair("1", 9), vpair("1", 3), vpair("1", 10),
       vpair("2", 3), vpair("3", 4), vpair("4", 3), vpair("4", 4),
       vpair("2", 6), vpair("3", 5), vpair("1", 7), vpair("5", 1)};
    std::size_t i = 0;
    for (vpair vp : connected_pairs) {
      g.add_edge(vp.first, vp.second, i);
      data[vp] = i;
    }
    ++i;
  }
  graph_type g;
  std::map<vpair,std::size_t> data;
};


BOOST_FIXTURE_TEST_CASE(test_edges, fixture) {
  for (edge_type e : g.edges()) {
    vpair vp(e.v1(), e.v2());
    BOOST_CHECK(data.count(vp) == 1);
    BOOST_CHECK_EQUAL(data[vp], g[e]);
    data.erase(vp);
  }
  BOOST_CHECK(data.empty());
}


BOOST_FIXTURE_TEST_CASE(test_neighbors, fixture) {
  std::vector<string> correct3 = {"1", "2", "4"}; // neighbors of 3
  std::vector<std::size_t> correct4 = {3, 4};          // neighbors of "4"

  std::multiset<string> actual3;
  std::multiset<std::size_t> actual4;
  boost::copy(g.neighbors(3),   std::inserter(actual3, actual3.begin()));
  boost::copy(g.neighbors("4"), std::inserter(actual4, actual4.begin()));

  BOOST_CHECK(boost::equal(correct3, actual3));
  BOOST_CHECK(boost::equal(correct4, actual4));
}


BOOST_FIXTURE_TEST_CASE(test_in_edges, fixture) {
  string v1 = "2";
  std::size_t v2 = 4;

  std::map<vpair, std::size_t> correct1;
  std::map<vpair, std::size_t> correct2;
  for (const auto& d : data) {
    if (d.first.first == v1) correct1.insert(d);
    if (d.first.second == v2) correct2.insert(d);
  }

  std::multimap<vpair, std::size_t> actual1;
  for (edge_type e : g.in_edges(v1)) {
    BOOST_CHECK(!e.forward());
    actual1.insert(std::make_pair(e.endpoints(), g[e]));
  }

  std::multimap<vpair, std::size_t> actual2;
  for (edge_type e : g.in_edges(v2)) {
    BOOST_CHECK(e.forward());
    actual2.insert(std::make_pair(e.endpoints(), g[e]));
  }

  BOOST_CHECK(boost::equal(correct1, actual1));
  BOOST_CHECK(boost::equal(correct2, actual2));
}


BOOST_FIXTURE_TEST_CASE(test_out_edges, fixture) {
  string v1 = "2";
  std::size_t v2 = 4;

  std::map<vpair, std::size_t> correct1;
  std::map<vpair, std::size_t> correct2;
  for (const auto& d : data) {
    if (d.first.first == v1) correct1.insert(d);
    if (d.first.second == v2) correct2.insert(d);
  }

  std::multimap<vpair, std::size_t> actual1;
  for (edge_type e : g.out_edges(v1)) {
    BOOST_CHECK(e.forward());
    actual1.insert(std::make_pair(e.endpoints(), g[e]));
  }

  std::multimap<vpair, std::size_t> actual2;
  for (edge_type e : g.out_edges(v2)) {
    BOOST_CHECK(!e.forward());
    actual2.insert(std::make_pair(e.endpoints(), g[e]));
  }

  BOOST_CHECK(boost::equal(correct1, actual1));
  BOOST_CHECK(boost::equal(correct2, actual2));
}


BOOST_FIXTURE_TEST_CASE(test_contains, fixture) {
  for (const auto& d : data) {
    string v1 = d.first.first;
    std::size_t v2 = d.first.second;
    BOOST_CHECK(g.contains(v1));
    BOOST_CHECK(g.contains(v2));
    BOOST_CHECK(g.contains(g.edge(v1, v2)));
    BOOST_CHECK(g.contains(g.edge(v2, v1)));
  }

  BOOST_CHECK(!g.contains(11));
  BOOST_CHECK(!g.contains("z"));
  BOOST_CHECK(!g.contains("0", 10));
  BOOST_CHECK(!g.contains("x", 20));
}


BOOST_FIXTURE_TEST_CASE(test_edge, fixture) {
  for (const auto& d : data) {
    string v1 = d.first.first;
    std::size_t v2 = d.first.second;
    BOOST_CHECK_EQUAL(g.edge(v1, v2).v1(), v1);
    BOOST_CHECK_EQUAL(g.edge(v1, v2).v2(), v2);
    BOOST_CHECK_EQUAL(g.edge(v1, v2).forward(), true);
    BOOST_CHECK_EQUAL(g.edge(v2, v1).v1(), v1);
    BOOST_CHECK_EQUAL(g.edge(v2, v1).v2(), v2);
    BOOST_CHECK_EQUAL(g.edge(v2, v1).forward(), false);
    BOOST_CHECK_EQUAL(g[g.edge(v1, v2)], d.second);
    BOOST_CHECK_EQUAL(g[g.edge(v2, v1)], d.second);
  }
}


BOOST_FIXTURE_TEST_CASE(test_degree, fixture) {
  std::map<string, std::size_t> degree1;
  std::map<std::size_t, std::size_t> degree2;
  for (const auto& d : data) {
    ++degree1[d.first.first];
    ++degree2[d.first.second];
  }

  for (string v1 : g.vertices1()) {
    BOOST_CHECK_EQUAL(g.degree(v1), degree1[v1]);
  }
  for (std::size_t v2 : g.vertices2()) {
    BOOST_CHECK_EQUAL(g.degree(v2), degree2[v2]);
  }
}


BOOST_FIXTURE_TEST_CASE(test_num, fixture) {
  BOOST_CHECK_EQUAL(g.num_vertices1(), 6);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 9);
  BOOST_CHECK_EQUAL(g.num_vertices(), 15);
  BOOST_CHECK_EQUAL(g.num_edges(), 12);
  g.remove_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices1(), 6);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 9);
  BOOST_CHECK_EQUAL(g.num_vertices(), 15);
  BOOST_CHECK_EQUAL(g.num_edges(), 9);
  g.remove_vertex(3);
  BOOST_CHECK_EQUAL(g.num_vertices1(), 6);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 8);
  BOOST_CHECK_EQUAL(g.num_vertices(), 14);
  BOOST_CHECK_EQUAL(g.num_edges(), 9);
  g.remove_vertex("1");
  BOOST_CHECK_EQUAL(g.num_vertices1(), 5);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 8);
  BOOST_CHECK_EQUAL(g.num_vertices(), 13);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);
}


BOOST_FIXTURE_TEST_CASE(test_comparison, fixture) {
  graph_type g1(g);
  BOOST_CHECK_EQUAL(g, g1);

  graph_type g2(g);
  g2[1] = 23;
  BOOST_CHECK_NE(g, g2);

  graph_type g3(g);
  g3.remove_edge("1", 3);
  g3.add_edge("1", 3, data[std::make_pair("1", 3)]);
  BOOST_CHECK_EQUAL(g, g3);

  graph_type g4(g);
  g4.remove_edge("1", 3);
  BOOST_CHECK_NE(g, g4);
}

BOOST_AUTO_TEST_CASE(test_serialization) {
  bipartite_graph<string, int, string, string, double> g;
  g.add_vertex("c", "maybe");
  g.add_vertex(1, "hello");
  g.add_vertex(2, "bye");
  g.add_edge("c", 1, 1.5);
  g.add_edge("c", 2, 2.5);
  BOOST_CHECK(serialize_deserialize(g));
}
