#define BOOST_TEST_MODULE cluster_graph
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/model/cluster_graph.hpp>

#include <unordered_set>
#include <utility>
#include <vector>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;
using Graph = ClusterGraph<Arg>;

struct VertexProperty {
  static int alive_count;
  int value = 0;

  explicit VertexProperty(int value = 0)
    : value(value) {
    ++alive_count;
  }

  VertexProperty(const VertexProperty& other)
    : value(other.value) {
    ++alive_count;
  }

  VertexProperty(VertexProperty&& other) noexcept
    : value(other.value) {
    ++alive_count;
  }

  VertexProperty& operator=(const VertexProperty&) = default;
  VertexProperty& operator=(VertexProperty&&) noexcept = default;

  ~VertexProperty() {
    --alive_count;
  }
};

int VertexProperty::alive_count = 0;

struct EdgeProperty {
  static int alive_count;
  int value = 0;

  explicit EdgeProperty(int value = 0)
    : value(value) {
    ++alive_count;
  }

  EdgeProperty(const EdgeProperty& other)
    : value(other.value) {
    ++alive_count;
  }

  EdgeProperty(EdgeProperty&& other) noexcept
    : value(other.value) {
    ++alive_count;
  }

  EdgeProperty& operator=(const EdgeProperty&) = default;
  EdgeProperty& operator=(EdgeProperty&&) noexcept = default;

  ~EdgeProperty() {
    --alive_count;
  }
};

int EdgeProperty::alive_count = 0;

struct Fixture {
  Fixture() {
    a = Arg("a");
    b = Arg("b");
    c = Arg("c");
    d = Arg("d");
    e = Arg("e");
    f = Arg("f");

    v1 = cg.add_vertex({a, b});
    v2 = cg.add_vertex({b, c, d});
    v3 = cg.add_vertex({c, d, e});
    v4 = cg.add_vertex({d, f});

    e12 = cg.add_edge(v1, v2);
    e23 = cg.add_edge(v2, v3);
    e24 = cg.add_edge(v2, v4);
  }

  using vertex_descriptor = Graph::vertex_descriptor;
  using edge_descriptor = Graph::edge_descriptor;

  Arg a;
  Arg b;
  Arg c;
  Arg d;
  Arg e;
  Arg f;

  Graph cg;
  vertex_descriptor v1 = nullptr;
  vertex_descriptor v2 = nullptr;
  vertex_descriptor v3 = nullptr;
  vertex_descriptor v4 = nullptr;
  edge_descriptor e12;
  edge_descriptor e23;
  edge_descriptor e24;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_copy_move) {
  Graph g1;
  BOOST_CHECK(g1.empty());

  Arg a("ca");
  Arg b("cb");
  auto* v1 = g1.add_vertex({a});
  auto* v2 = g1.add_vertex({a, b});
  auto e12 = g1.add_edge(v1, v2);

  BOOST_CHECK(g1.contains(v1));
  BOOST_CHECK(g1.contains(v2));
  BOOST_CHECK(g1.contains(e12));

  Graph g2(g1);
  BOOST_CHECK_EQUAL(g2.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g2.num_edges(), 1);
  BOOST_CHECK(g2.is_connected());
  BOOST_CHECK(g2.is_tree());
  BOOST_CHECK(g2.has_running_intersection());

  Graph g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g3.num_edges(), 1);

  Graph g4(std::move(g3));
  BOOST_CHECK_EQUAL(g4.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g4.num_edges(), 1);

  Graph g5;
  g5 = std::move(g4);
  BOOST_CHECK_EQUAL(g5.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g5.num_edges(), 1);
}

BOOST_AUTO_TEST_CASE(base_cluster_graph_uses_annotated_properties) {
  Graph cg;
  Arg a("prop_a");
  Arg b("prop_b");

  auto* v1 = cg.add_vertex({a});
  auto* v2 = cg.add_vertex({a, b});
  auto e = cg.add_edge(v1, v2);

  BOOST_CHECK(cg.property(v1).type_info == typeid(Annotated<IndexedDomain<Graph::Vertex, Arg>, void>));
  BOOST_CHECK(cg.property(v1).ptr != nullptr);
  BOOST_CHECK(cg.property(e).type_info == typeid(Annotated<IndexedDomain<Graph::Edge, Arg>, void>));
  BOOST_CHECK(cg.property(e).ptr != nullptr);
}

BOOST_FIXTURE_TEST_CASE(test_domain_accessors_and_queries, Fixture) {
  BOOST_CHECK_EQUAL(cg.num_arguments(), 6);
  BOOST_CHECK_EQUAL(cg.count(a), 1);
  BOOST_CHECK_EQUAL(cg.count(d), 3);
  BOOST_CHECK_EQUAL(cg.count(f), 1);
  BOOST_CHECK_EQUAL(cg.count(Arg("not_present")), 0);

  BOOST_CHECK(cg.cluster(v2) == libgm::Domain<Arg>({b, c, d}));
  BOOST_CHECK(cg.separator(e12) == libgm::Domain<Arg>({b}));
  BOOST_CHECK(cg.separator(e23) == libgm::Domain<Arg>({c, d}));
  BOOST_CHECK(cg.separator(e24) == libgm::Domain<Arg>({d}));

  BOOST_CHECK(cg.is_connected());
  BOOST_CHECK(cg.is_tree());
  BOOST_CHECK(cg.has_running_intersection());
  BOOST_CHECK(cg.is_triangulated());

  MarkovNetwork<Arg> mn = cg.markov_network();
  BOOST_CHECK_EQUAL(mn.num_vertices(), 6);
  BOOST_CHECK(mn.contains(a, b));
  BOOST_CHECK(mn.contains(b, c));
  BOOST_CHECK(mn.contains(b, d));
  BOOST_CHECK(mn.contains(c, d));
  BOOST_CHECK(mn.contains(c, e));
  BOOST_CHECK(mn.contains(d, e));
  BOOST_CHECK(mn.contains(d, f));
}

BOOST_FIXTURE_TEST_CASE(test_updates_and_removals, Fixture) {
  cg.update_cluster(v4, {e, f});
  BOOST_CHECK(cg.cluster(v4) == libgm::Domain<Arg>({e, f}));
  BOOST_CHECK(!cg.find_cluster_cover({d, f}));

  cg.update_separator(e24, {});
  BOOST_CHECK(cg.separator(e24) == libgm::Domain<Arg>());
  std::unordered_set<edge_descriptor> separators_with_d = {e23};
  cg.intersecting_separators({d}, [&](edge_descriptor e) {
    BOOST_CHECK_EQUAL(separators_with_d.erase(e), 1);
  });
  BOOST_CHECK(separators_with_d.empty());

  BOOST_CHECK_EQUAL(cg.remove_edge(v1, v3), 0);
  BOOST_CHECK_EQUAL(cg.remove_edge(v1, v2), 1);
  BOOST_CHECK_EQUAL(cg.num_edges(), 2);
  BOOST_CHECK(!cg.find_separator_cover({b}));
  BOOST_CHECK(cg.find_separator_cover({c, d}));

  cg.clear_vertex(v2);
  BOOST_CHECK_EQUAL(cg.num_edges(), 0);
  BOOST_CHECK(!cg.find_separator_cover({d}));

  cg.remove_vertex(v1);
  BOOST_CHECK_EQUAL(cg.num_vertices(), 3);
  BOOST_CHECK(!cg.find_cluster_cover({a, b}));

  cg.clear();
  BOOST_CHECK(cg.empty());
}

BOOST_FIXTURE_TEST_CASE(test_find_and_intersection_queries, Fixture) {
  auto* cover_ab = cg.find_cluster_cover({a, b});
  BOOST_CHECK(cover_ab != nullptr);
  BOOST_CHECK(cg.cluster(cover_ab) == libgm::Domain<Arg>({a, b}));

  auto sep_d = cg.find_separator_cover({d});
  BOOST_CHECK(cg.contains(sep_d));
  BOOST_CHECK(cg.separator(sep_d) == libgm::Domain<Arg>({d}));

  auto* meets_f = cg.find_cluster_meets({f});
  BOOST_CHECK(meets_f != nullptr);
  BOOST_CHECK(cg.cluster(meets_f) == libgm::Domain<Arg>({d, f}));

  auto sep_b = cg.find_separator_meets({b});
  BOOST_CHECK(cg.contains(sep_b));
  BOOST_CHECK(cg.separator(sep_b) == libgm::Domain<Arg>({b}));

  std::unordered_set<Fixture::vertex_descriptor> clusters_with_d = {v2, v3, v4};
  cg.intersecting_clusters({d}, [&](Fixture::vertex_descriptor v) {
    BOOST_CHECK_EQUAL(clusters_with_d.erase(v), 1);
  });
  BOOST_CHECK(clusters_with_d.empty());

  std::unordered_set<Fixture::edge_descriptor> seps_with_d = {e23, e24};
  cg.intersecting_separators({d}, [&](Fixture::edge_descriptor e) {
    BOOST_CHECK_EQUAL(seps_with_d.erase(e), 1);
  });
  BOOST_CHECK(seps_with_d.empty());
}

BOOST_FIXTURE_TEST_CASE(test_merge, Fixture) {
  Fixture::vertex_descriptor retained = cg.merge(e12);
  BOOST_CHECK(retained != nullptr);
  BOOST_CHECK_EQUAL(cg.degree(retained), 2);
  BOOST_CHECK_EQUAL(cg.num_vertices(), 3);
  BOOST_CHECK_EQUAL(cg.num_edges(), 2);
  BOOST_CHECK(cg.cluster(retained) == libgm::Domain<Arg>({a, b, c, d}));
}

BOOST_AUTO_TEST_CASE(test_triangulated_from_cliques) {
  Arg a("ta");
  Arg b("tb");
  Arg c("tc");
  Arg d("td");
  Arg e("te");

  Graph jt;
  std::vector<Graph::vertex_descriptor> v =
    jt.triangulated({libgm::Domain<Arg>({a, b, c}), libgm::Domain<Arg>({b, c, d}), libgm::Domain<Arg>({c, d, e})});

  BOOST_CHECK_EQUAL(jt.num_vertices(), 3);
  BOOST_CHECK_EQUAL(jt.num_edges(), 2);
  BOOST_CHECK(jt.contains(v[0], v[1]));
  BOOST_CHECK(jt.contains(v[1], v[2]));
  BOOST_CHECK(jt.is_connected());
  BOOST_CHECK(jt.is_tree());
  BOOST_CHECK(jt.has_running_intersection());
  BOOST_CHECK(jt.is_triangulated());
}

BOOST_AUTO_TEST_CASE(test_triangulated_from_markov_network) {
  Arg a("a");
  Arg b("b");
  Arg c("c");
  Arg d("d");
  Arg e("e");
  Arg f("f");
  std::vector<Arg> vertices = {a, b, c, d, e, f};
  std::vector<std::pair<Arg, Arg>> edges = {{a, b}, {b, c}, {c, d}, {a, d}, {d, e}, {d, f}, {e, f}};

  MarkovNetwork<Arg> mn;
  for (Arg v : vertices) {
    mn.add_vertex(v);
  }
  for (auto [u, v] : edges) {
    mn.add_edge(u, v);
  }

  Graph jt;
  MarkovStructure mg = mn.structure();
  jt.triangulated(mg, MinDegreeStrategy());

  BOOST_CHECK(jt.num_vertices() > 0);
  BOOST_CHECK_EQUAL(jt.num_edges(), jt.num_vertices() - 1);
  BOOST_CHECK(jt.is_connected());
  BOOST_CHECK(jt.is_tree());
  BOOST_CHECK(jt.has_running_intersection());
  BOOST_CHECK(jt.is_triangulated());
}

BOOST_AUTO_TEST_CASE(test_typed_properties) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  ClusterGraph<Arg, VertexProperty, EdgeProperty> cg;
  Arg a("typed_a");
  Arg b("typed_b");
  Arg c("typed_c");

  auto* v1 = cg.add_vertex({a, b}, VertexProperty(10));
  auto* v2 = cg.add_vertex({b, c}, VertexProperty(20));
  auto e = cg.add_edge(v1, v2, {b}, EdgeProperty(30));

  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(cg[v1].value, 10);
  BOOST_CHECK_EQUAL(cg[v2].value, 20);
  BOOST_CHECK_EQUAL(cg[e].value, 30);

  cg.clear();
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);
}
