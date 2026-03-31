#define BOOST_TEST_MODULE markov_network
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/model/markov_network.hpp>

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

struct VertexProperty {
  int value = 0;

  explicit VertexProperty(int v = 0)
    : value(v) {}
};

struct EdgeProperty {
  int value = 0;

  explicit EdgeProperty(int v = 0)
    : value(v) {}
};

void add_vertices(MarkovNetwork<void>& g, const std::vector<Arg>& vertices) {
  for (Arg v : vertices) {
    g.add_vertex(v);
  }
}

void add_edges(MarkovNetwork<void>& g, const std::vector<std::pair<Arg, Arg>>& edges) {
  for (auto [u, v] : edges) {
    if (v < u) {
      std::swap(u, v);
    }
    auto [e, inserted] = g.add_edge(u, v);
    BOOST_CHECK(inserted);
    BOOST_CHECK(g.contains(e));
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(test_edge_descriptor_basics) {
  using edge_type = MarkovNetwork<void>::edge_descriptor;
  MarkovNetwork g;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  add_vertices(g, {a, b});
  auto [e, inserted] = g.add_edge(a, b);
  BOOST_CHECK(inserted);

  edge_type empty;
  BOOST_CHECK(!empty);

  BOOST_CHECK(e);
  BOOST_CHECK_EQUAL(g.argument(e.source()), a);
  BOOST_CHECK_EQUAL(g.argument(e.target()), b);
  BOOST_CHECK_EQUAL(g.argument(e.reverse().source()), b);
  BOOST_CHECK_EQUAL(g.argument(e.reverse().target()), a);
  BOOST_CHECK_EQUAL(g.domain(e), Domain({a, b}));
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
  for (auto* v : g.vertices()) {
    BOOST_CHECK_EQUAL(expected.erase(g.argument(v)), 1);
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
  for (auto* v : g.adjacent_vertices(c)) {
    BOOST_CHECK_EQUAL(neighbors.erase(g.argument(v)), 1);
  }
  BOOST_CHECK(neighbors.empty());
}

BOOST_AUTO_TEST_CASE(test_in_and_out_edges) {
  using edge_type = MarkovNetwork<void>::edge_descriptor;

  MarkovNetwork g;
  Arg c = make_arg("c");
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg d = make_arg("d");
  Arg e = make_arg("e");
  add_vertices(g, {a, b, c, d, e});
  add_edges(g, {{c, a}, {c, b}, {c, d}, {e, c}});

  std::unordered_set<Domain> expected_out = {
    Domain({c, a}), Domain({c, b}), Domain({c, d}), Domain({c, e})
  };
  for (edge_type oe : g.out_edges(c)) {
    BOOST_CHECK_EQUAL(expected_out.erase(g.domain(oe)), 1);
  }
  BOOST_CHECK(expected_out.empty());

  std::unordered_set<Domain> expected_in = {
    Domain({a, c}), Domain({b, c}), Domain({d, c}), Domain({e, c})
  };
  for (edge_type ie : g.in_edges(c)) {
    BOOST_CHECK_EQUAL(expected_in.erase(g.domain(ie)), 1);
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
  BOOST_CHECK_EQUAL(g.argument(eab.source()), a);
  BOOST_CHECK_EQUAL(g.argument(eab.target()), b);
  BOOST_CHECK_EQUAL(g.argument(eba.source()), b);
  BOOST_CHECK_EQUAL(g.argument(eba.target()), a);
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

BOOST_AUTO_TEST_CASE(test_add_edge_requires_canonical_order) {
  MarkovNetwork mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK_THROW(mn.add_edge(b, a), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_add_edge_with_property_requires_canonical_order) {
  MarkovNetwork<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK_THROW(mn.add_edge(b, a, EdgeProperty(7)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_untyped_edge_property_is_null) {
  MarkovNetwork mn;
  Arg a = make_arg("typed_a");
  Arg b = make_arg("typed_b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));

  auto [e, inserted] = mn.add_edge(a, b);
  BOOST_CHECK(inserted);
  BOOST_CHECK_EQUAL(mn.property(e).ptr, nullptr);
}

BOOST_AUTO_TEST_CASE(test_typed_vertex_and_edge_properties) {
  MarkovNetwork<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("typed_a");
  Arg b = make_arg("typed_b");

  BOOST_CHECK(mn.add_vertex(a, VertexProperty(10)));
  BOOST_CHECK(mn.add_vertex(b, VertexProperty(20)));
  BOOST_CHECK_EQUAL(mn[a].value, 10);
  BOOST_CHECK_EQUAL(mn[b].value, 20);

  auto [e, inserted] = mn.add_edge(a, b, EdgeProperty(30));
  BOOST_CHECK(inserted);
  BOOST_CHECK_EQUAL(mn[e].value, 30);
}

BOOST_AUTO_TEST_CASE(test_typed_default_constructed_properties_and_clear) {
  MarkovNetwork<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("default_a");
  Arg b = make_arg("default_b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));
  auto [e, inserted] = mn.add_edge(a, b);
  BOOST_CHECK(inserted);

  BOOST_CHECK_EQUAL(mn[a].value, 0);
  BOOST_CHECK_EQUAL(mn[b].value, 0);
  BOOST_CHECK_EQUAL(mn[e].value, 0);

  mn.clear();
  BOOST_CHECK(mn.empty());
}

BOOST_AUTO_TEST_CASE(test_typed_add_edge_does_not_overwrite_existing_property) {
  MarkovNetwork<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("edge_a");
  Arg b = make_arg("edge_b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));

  auto [e1, inserted1] = mn.add_edge(a, b, EdgeProperty(1));
  BOOST_CHECK(inserted1);
  BOOST_CHECK_EQUAL(mn[e1].value, 1);

  auto [e2, inserted2] = mn.add_edge(a, b, EdgeProperty(9));
  BOOST_CHECK(!inserted2);
  BOOST_CHECK_EQUAL(mn[e2].value, 1);
}

BOOST_AUTO_TEST_CASE(test_typed_init_vertex_and_edge_properties) {
  MarkovNetwork<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("init_a");
  Arg b = make_arg("init_b");
  Arg c = make_arg("init_c");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));
  BOOST_CHECK(mn.add_vertex(c));
  BOOST_CHECK(mn.add_edge(a, b).second);
  BOOST_CHECK(mn.add_edge(b, c).second);

  int vertex_calls = 0;
  mn.init_vertices([&](Arg u) {
    ++vertex_calls;
    if (u == a) return VertexProperty(11);
    if (u == b) return VertexProperty(22);
    return VertexProperty(33);
  });

  BOOST_CHECK_EQUAL(vertex_calls, 3);
  BOOST_CHECK_EQUAL(mn[a].value, 11);
  BOOST_CHECK_EQUAL(mn[b].value, 22);
  BOOST_CHECK_EQUAL(mn[c].value, 33);

  int edge_calls = 0;
  mn.init_edges([&](auto e) {
    ++edge_calls;
    if (mn.domain(e) == Domain({a, b})) {
      return EdgeProperty(44);
    }
    return EdgeProperty(55);
  });

  BOOST_CHECK_EQUAL(edge_calls, 2);
  BOOST_CHECK_EQUAL(mn[mn.edge(a, b)].value, 44);
  BOOST_CHECK_EQUAL(mn[mn.edge(b, c)].value, 55);
}
