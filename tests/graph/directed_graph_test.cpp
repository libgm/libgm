#define BOOST_TEST_MODULE directed_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/directed_graph.hpp>

#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../predicates.hpp"

using namespace libgm;

namespace {

template <typename VP = void>
struct TestDirectedGraph : DirectedGraph {
  TestDirectedGraph()
    : DirectedGraph(property_layout<VP>()) {}

  using DirectedGraph::add_vertex;
  using DirectedGraph::property;

  using property_reference = std::add_lvalue_reference_t<VP>;
  using const_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;

  property_reference operator[](Vertex* u) {
    if constexpr (std::is_void_v<VP>) {
      return;
    } else {
      return opaque_cast<VP>(property(u));
    }
  }

  const_property_reference operator[](Vertex* u) const {
    if constexpr (std::is_void_v<VP>) {
      return;
    } else {
      return opaque_cast<VP>(property(u));
    }
  }
};

struct CountingProperty {
  static int alive_count;
  int value = 0;

  explicit CountingProperty(int v = 0)
    : value(v) {
    ++alive_count;
  }

  CountingProperty(const CountingProperty& other)
    : value(other.value) {
    ++alive_count;
  }

  CountingProperty(CountingProperty&& other) noexcept
    : value(other.value) {
    ++alive_count;
  }

  CountingProperty& operator=(const CountingProperty& other) = default;
  CountingProperty& operator=(CountingProperty&& other) noexcept = default;

  ~CountingProperty() {
    --alive_count;
  }
};

int CountingProperty::alive_count = 0;

struct Fixture {
  Fixture() {
    v0 = g.add_vertex();
    v1 = g.add_vertex();
    v2 = g.add_vertex({v1});
    v3 = g.add_vertex({v1, v2});
    v4 = g.add_vertex({v0, v3});
  }

  TestDirectedGraph<> g;
  DirectedGraph::Vertex* v0 = nullptr;
  DirectedGraph::Vertex* v1 = nullptr;
  DirectedGraph::Vertex* v2 = nullptr;
  DirectedGraph::Vertex* v3 = nullptr;
  DirectedGraph::Vertex* v4 = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_empty) {
  TestDirectedGraph<> g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK_EQUAL(g1.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g1.num_edges(), 0);
  BOOST_CHECK(g1.vertices().empty());

  auto* a = g1.add_vertex();
  auto* b = g1.add_vertex({a});

  TestDirectedGraph<> g2(g1);
  BOOST_CHECK(!g2.empty());
  BOOST_CHECK_EQUAL(g2.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g2.num_edges(), 1);

  TestDirectedGraph<> g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g3.num_edges(), 1);

  TestDirectedGraph<> g4(std::move(g3));
  BOOST_CHECK_EQUAL(g4.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g4.num_edges(), 1);

  TestDirectedGraph<> g5;
  g5 = std::move(g4);
  BOOST_CHECK_EQUAL(g5.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g5.num_edges(), 1);

  BOOST_CHECK(g1.contains(a));
  BOOST_CHECK(g1.contains(a, b));
}

BOOST_FIXTURE_TEST_CASE(test_edges_in_out_and_adjacency, Fixture) {
  using edge_type = DirectedGraph::edge_descriptor;

  BOOST_CHECK(g.contains(v1, v2));
  BOOST_CHECK(g.contains(v1, v3));
  BOOST_CHECK(g.contains(v2, v3));
  BOOST_CHECK(g.contains(v0, v4));
  BOOST_CHECK(g.contains(v3, v4));
  BOOST_CHECK(!g.contains(v4, v3));

  edge_type e = g.edge(v1, v3);
  BOOST_CHECK(g.contains(e));
  BOOST_CHECK_EQUAL(e.source(), v1);
  BOOST_CHECK_EQUAL(e.target(), v3);
  BOOST_CHECK(!g.edge(v4, v3));
  BOOST_CHECK(!g.edge(v1, v4));

  std::vector<edge_type> expected_in3 = {{v1, v3}, {v2, v3}};
  BOOST_CHECK(range_equal(g.in_edges(v3), expected_in3));

  std::vector<edge_type> expected_out1 = {{v1, v2}, {v1, v3}};
  BOOST_CHECK(range_equal(g.out_edges(v1), expected_out1));

  std::vector<DirectedGraph::Vertex*> expected_children1 = {v2, v3};
  BOOST_CHECK(range_equal(g.adjacent_vertices(v1), expected_children1));
}

BOOST_AUTO_TEST_CASE(test_vertex_and_parent_validation) {
  TestDirectedGraph<> g;
  auto* a = g.add_vertex();
  auto* b = g.add_vertex();

  BOOST_CHECK_THROW(g.add_vertex({nullptr}), std::out_of_range);
  BOOST_CHECK_THROW(g.set_parents(a, {a}), std::invalid_argument);
  BOOST_CHECK_THROW(g.set_parents(b, {nullptr}), std::out_of_range);
}

BOOST_FIXTURE_TEST_CASE(test_parents_degree_and_counts, Fixture) {
  BOOST_CHECK(g.parents(v0).empty());
  BOOST_CHECK(range_equal(g.parents(v2), std::vector{v1}));
  BOOST_CHECK(range_equal(g.parents(v3), std::vector{v1, v2}));
  BOOST_CHECK(range_equal(g.parents(v4), std::vector{v0, v3}));

  BOOST_CHECK_EQUAL(g.in_degree(v3), 2);
  BOOST_CHECK_EQUAL(g.out_degree(v1), 2);
  BOOST_CHECK_EQUAL(g.degree(v3), 3);

  BOOST_CHECK_EQUAL(g.num_vertices(), 5);
  BOOST_CHECK_EQUAL(g.num_edges(), 5);
}

BOOST_FIXTURE_TEST_CASE(test_remove_and_clear, Fixture) {
  BOOST_CHECK_EQUAL(g.remove_vertex(v4), 1);
  BOOST_CHECK_EQUAL(g.num_vertices(), 4);
  BOOST_CHECK_EQUAL(g.num_edges(), 3);

  g.remove_in_edges(v3);
  BOOST_CHECK(g.parents(v3).empty());
  BOOST_CHECK_EQUAL(g.in_degree(v3), 0);
  BOOST_CHECK_EQUAL(g.out_degree(v1), 1);
  BOOST_CHECK_EQUAL(g.out_degree(v2), 0);
  BOOST_CHECK_EQUAL(g.num_edges(), 1);
  BOOST_CHECK(!g.contains(v1, v3));
  BOOST_CHECK(!g.contains(v2, v3));

  g.clear();
  BOOST_CHECK(g.empty());
  BOOST_CHECK_EQUAL(g.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g.num_edges(), 0);
}

BOOST_FIXTURE_TEST_CASE(test_remove_vertex_preconditions, Fixture) {
  BOOST_CHECK_THROW(g.remove_vertex(v1), std::logic_error);
  BOOST_CHECK_EQUAL(g.remove_vertex(nullptr), 0);
}

BOOST_FIXTURE_TEST_CASE(test_remove_in_edges_missing_vertex_throws, Fixture) {
  BOOST_CHECK_THROW(g.remove_in_edges(nullptr), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_untyped_property_layout) {
  TestDirectedGraph<> g;
  auto* v = g.add_vertex();

  BOOST_CHECK(g.property(v).type_info == typeid(void));
  BOOST_CHECK_EQUAL(g.property(v).ptr, nullptr);
}

BOOST_AUTO_TEST_CASE(test_typed_property_handling) {
  CountingProperty::alive_count = 0;

  TestDirectedGraph<CountingProperty> g;
  auto* a = g.add_vertex();
  auto* b = g.add_vertex({a});

  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 2);
  g[a] = CountingProperty(7);
  g[b] = CountingProperty(9);
  BOOST_CHECK_EQUAL(g[a].value, 7);
  BOOST_CHECK_EQUAL(g[b].value, 9);
  BOOST_CHECK_EQUAL(static_cast<void*>(&g[a]), g.property(a).ptr);

  g.remove_vertex(b);
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 1);

  g.clear();
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}
