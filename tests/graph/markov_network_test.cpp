#define BOOST_TEST_MODULE markov_network
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/markov_network.hpp>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

Arg make_arg(size_t i) {
  return NamedFactory::default_factory().make("v" + std::to_string(i));
}

void add_vertices(MarkovNetwork& g, const std::vector<Arg>& vertices) {
  for (Arg v : vertices) {
    g.add_vertex(v);
  }
}

void add_edges(MarkovNetwork& g, const std::vector<std::pair<Arg, Arg>>& edges) {
  for (auto [u, v] : edges) {
    auto [e, inserted] = g.add_edge(u, v);
    BOOST_CHECK(inserted);
    BOOST_CHECK(g.contains(e));
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(test_edge_descriptor_basics) {
  using edge_type = MarkovNetwork::edge_descriptor;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  edge_type empty;
  BOOST_CHECK(!empty);

  edge_type e(a, b);
  BOOST_CHECK(e);
  BOOST_CHECK_EQUAL(e.source(), a);
  BOOST_CHECK_EQUAL(e.target(), b);
  BOOST_CHECK_EQUAL(e.reverse().source(), b);
  BOOST_CHECK_EQUAL(e.reverse().target(), a);
  BOOST_CHECK(e.unordered_pair() == std::minmax(a, b));
}

BOOST_AUTO_TEST_CASE(test_constructors) {
  MarkovNetwork g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK(g1.vertices().empty());
  BOOST_CHECK_EQUAL(g1.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g1.num_edges(), 0);

  std::vector<Arg> vertices;
  for (size_t i = 1; i <= 7; ++i) {
    vertices.push_back(make_arg(i));
  }

  std::vector<std::pair<Arg, Arg>> edges = {
    {vertices[0], vertices[1]}, {vertices[0], vertices[2]},
    {vertices[0], vertices[3]}, {vertices[1], vertices[2]},
    {vertices[2], vertices[4]}, {vertices[4], vertices[5]}
  };

  MarkovNetwork g2;
  add_vertices(g2, vertices);
  add_edges(g2, edges);

  for (auto [u, v] : edges) {
    BOOST_CHECK(g2.contains(u));
    BOOST_CHECK(g2.contains(v));
    BOOST_CHECK(g2.contains(u, v));
    BOOST_CHECK(g2.contains(v, u));
  }

  MarkovNetwork g3(g2);
  for (auto [u, v] : edges) {
    BOOST_CHECK(g3.contains(u));
    BOOST_CHECK(g3.contains(v));
    BOOST_CHECK(g3.contains(u, v));
    BOOST_CHECK(g3.contains(v, u));
  }
  BOOST_CHECK_EQUAL(g3.num_vertices(), g2.num_vertices());
  BOOST_CHECK_EQUAL(g3.num_edges(), g2.num_edges());

  MarkovNetwork g4;
  g4 = g2;
  BOOST_CHECK_EQUAL(g4.num_vertices(), g2.num_vertices());
  BOOST_CHECK_EQUAL(g4.num_edges(), g2.num_edges());

  MarkovNetwork g5(std::move(g4));
  BOOST_CHECK_EQUAL(g5.num_vertices(), g2.num_vertices());
  BOOST_CHECK_EQUAL(g5.num_edges(), g2.num_edges());

  MarkovNetwork g6;
  g6 = std::move(g5);
  BOOST_CHECK_EQUAL(g6.num_vertices(), g2.num_vertices());
  BOOST_CHECK_EQUAL(g6.num_edges(), g2.num_edges());
}

BOOST_AUTO_TEST_CASE(test_vertices) {
  MarkovNetwork g;
  std::vector<Arg> vertices;
  for (size_t i = 1; i <= 10; ++i) {
    Arg v = make_arg(i);
    vertices.push_back(v);
    BOOST_CHECK(g.add_vertex(v));
  }

  std::unordered_set<Arg> expected(vertices.begin(), vertices.end());
  for (Arg v : g.vertices()) {
    BOOST_CHECK_EQUAL(expected.erase(v), 1);
  }
  BOOST_CHECK(expected.empty());
}

BOOST_AUTO_TEST_CASE(test_adjacent_vertices) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg d = make_arg("d");
  Arg e = make_arg("e");
  add_vertices(g, {a, b, c, d, e});
  add_edges(g, {{c, a}, {c, b}, {c, d}, {e, c}});

  std::unordered_set<Arg> neighbors = {a, b, d, e};
  for (Arg v : g.adjacent_vertices(c)) {
    BOOST_CHECK_EQUAL(neighbors.erase(v), 1);
  }
  BOOST_CHECK(neighbors.empty());
}

BOOST_AUTO_TEST_CASE(test_in_and_out_edges) {
  using edge_type = MarkovNetwork::edge_descriptor;

  MarkovNetwork g;
  Arg c = make_arg("c");
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg d = make_arg("d");
  Arg e = make_arg("e");
  add_vertices(g, {a, b, c, d, e});
  add_edges(g, {{c, a}, {c, b}, {c, d}, {e, c}});

  std::unordered_set<edge_type> expected_out = {{c, a}, {c, b}, {c, d}, {c, e}};
  for (edge_type oe : g.out_edges(c)) {
    BOOST_CHECK_EQUAL(expected_out.erase(oe), 1);
  }
  BOOST_CHECK(expected_out.empty());

  std::unordered_set<edge_type> expected_in = {{a, c}, {b, c}, {d, c}, {e, c}};
  for (edge_type ie : g.in_edges(c)) {
    BOOST_CHECK_EQUAL(expected_in.erase(ie), 1);
  }
  BOOST_CHECK(expected_in.empty());
}

BOOST_AUTO_TEST_CASE(test_contains_and_edge) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  add_vertices(g, {a, b, c});
  add_edges(g, {{a, b}, {b, c}});

  BOOST_CHECK(g.contains(a));
  BOOST_CHECK(g.contains(b));
  BOOST_CHECK(g.contains(c));

  BOOST_CHECK(g.contains(a, b));
  BOOST_CHECK(g.contains(b, a));
  BOOST_CHECK(g.contains(b, c));
  BOOST_CHECK(g.contains(c, b));
  BOOST_CHECK(!g.contains(a, c));
  BOOST_CHECK(!g.contains(c, a));

  auto eab = g.edge(a, b);
  auto eba = g.edge(b, a);
  BOOST_CHECK_EQUAL(eab.source(), a);
  BOOST_CHECK_EQUAL(eab.target(), b);
  BOOST_CHECK_EQUAL(eba.source(), b);
  BOOST_CHECK_EQUAL(eba.target(), a);
  BOOST_CHECK(g.contains(eab));
  BOOST_CHECK(g.contains(eba));
}

BOOST_AUTO_TEST_CASE(test_degree) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg d = make_arg("d");
  add_vertices(g, {a, b, c, d});
  add_edges(g, {{a, b}, {a, c}, {d, a}});

  BOOST_CHECK_EQUAL(g.in_degree(a), g.out_degree(a));
  BOOST_CHECK_EQUAL(g.out_degree(a), g.degree(a));
  BOOST_CHECK_EQUAL(g.degree(a), 3);
  BOOST_CHECK_EQUAL(g.degree(b), 1);
}

BOOST_AUTO_TEST_CASE(test_num_and_removals) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg d = make_arg("d");
  Arg e = make_arg("e");
  add_vertices(g, {a, b, c, d, e});
  add_edges(g, {{a, b}, {a, c}, {a, d}, {b, c}, {d, e}});

  BOOST_CHECK_EQUAL(g.num_vertices(), 5);
  BOOST_CHECK_EQUAL(g.num_edges(), 5);

  g.remove_edges(a);
  BOOST_CHECK_EQUAL(g.num_vertices(), 5);
  BOOST_CHECK_EQUAL(g.num_edges(), 2);
  BOOST_CHECK_EQUAL(g.degree(a), 0);

  BOOST_CHECK_EQUAL(g.remove_edge(d, e), 1);
  BOOST_CHECK_EQUAL(g.num_edges(), 1);

  BOOST_CHECK_EQUAL(g.remove_vertex(a), 1);
  BOOST_CHECK_EQUAL(g.num_vertices(), 4);

  g.clear();
  BOOST_CHECK(g.empty());
  BOOST_CHECK_EQUAL(g.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g.num_edges(), 0);
}

BOOST_AUTO_TEST_CASE(test_remove_returns_zero_on_missing) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  add_vertices(g, {a, b});
  add_edges(g, {{a, b}});

  BOOST_CHECK_EQUAL(g.remove_edge(a, c), 0);
  BOOST_CHECK_EQUAL(g.remove_edge(c, a), 0);
  BOOST_CHECK_EQUAL(g.remove_edge(a, b), 1);
  BOOST_CHECK_EQUAL(g.remove_edge(a, b), 0);

  BOOST_CHECK_EQUAL(g.remove_vertex(c), 0);
  BOOST_CHECK_EQUAL(g.remove_vertex(a), 1);
  BOOST_CHECK_EQUAL(g.remove_vertex(a), 0);
}

BOOST_AUTO_TEST_CASE(test_add_clique_and_add_edges) {
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg d = make_arg("d");

  add_vertices(g, {a, b, c, d});

  g.add_edges(a, {b, c});
  BOOST_CHECK(g.contains(a, b));
  BOOST_CHECK(g.contains(a, c));

  g.add_clique({b, c, d});
  BOOST_CHECK(g.contains(b, c));
  BOOST_CHECK(g.contains(b, d));
  BOOST_CHECK(g.contains(c, d));

  BOOST_CHECK_EQUAL(g.num_vertices(), 4);
  BOOST_CHECK_EQUAL(g.num_edges(), 5);
}
