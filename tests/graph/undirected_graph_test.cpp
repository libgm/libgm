#define BOOST_TEST_MODULE undirected_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/undirected_graph.hpp>

#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace libgm;

namespace {

template <typename VP = void, typename EP = void>
struct TestUndirectedGraph : UndirectedGraph {
  TestUndirectedGraph()
    : UndirectedGraph(property_layout<VP>(), property_layout<EP>()) {}

  using UndirectedGraph::add_edge;
  using UndirectedGraph::add_vertex;
  using UndirectedGraph::property;

  using vertex_property_reference = std::add_lvalue_reference_t<VP>;
  using const_vertex_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;
  using edge_property_reference = std::add_lvalue_reference_t<EP>;
  using const_edge_property_reference = std::add_lvalue_reference_t<std::add_const_t<EP>>;

  vertex_property_reference vertex_property(Vertex* v) {
    if constexpr (std::is_void_v<VP>) {
      return;
    } else {
      return opaque_cast<VP>(property(v));
    }
  }

  edge_property_reference edge_property(edge_descriptor e) {
    if constexpr (std::is_void_v<EP>) {
      return;
    } else {
      return opaque_cast<EP>(property(e));
    }
  }
};

struct CountingProperty {
  static int alive_count;
  int value = 0;

  explicit CountingProperty(int value = 0)
    : value(value) {
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

  CountingProperty& operator=(const CountingProperty&) = default;
  CountingProperty& operator=(CountingProperty&&) noexcept = default;

  ~CountingProperty() {
    --alive_count;
  }
};

int CountingProperty::alive_count = 0;

struct Fixture {
  Fixture() {
    v1 = g.add_vertex();
    v2 = g.add_vertex();
    v3 = g.add_vertex();
    v4 = g.add_vertex();

    e12 = g.add_edge(v1, v2);
    e23 = g.add_edge(v2, v3);
    e24 = g.add_edge(v2, v4);
  }

  TestUndirectedGraph<> g;
  UndirectedGraph::Vertex* v1 = nullptr;
  UndirectedGraph::Vertex* v2 = nullptr;
  UndirectedGraph::Vertex* v3 = nullptr;
  UndirectedGraph::Vertex* v4 = nullptr;
  UndirectedGraph::edge_descriptor e12;
  UndirectedGraph::edge_descriptor e23;
  UndirectedGraph::edge_descriptor e24;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_copy_move) {
  TestUndirectedGraph<> g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK_EQUAL(g1.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g1.num_edges(), 0);

  auto* v1 = g1.add_vertex();
  auto* v2 = g1.add_vertex();
  auto e12 = g1.add_edge(v1, v2);

  BOOST_CHECK(g1.contains(v1));
  BOOST_CHECK(g1.contains(v2));
  BOOST_CHECK(g1.contains(e12));
  BOOST_CHECK_EQUAL(g1.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g1.num_edges(), 1);

  TestUndirectedGraph<> g2(g1);
  BOOST_CHECK_EQUAL(g2.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g2.num_edges(), 1);
  BOOST_CHECK(g2.is_connected());
  BOOST_CHECK(g2.is_tree());

  TestUndirectedGraph<> g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g3.num_edges(), 1);

  TestUndirectedGraph<> g4(std::move(g3));
  BOOST_CHECK_EQUAL(g4.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g4.num_edges(), 1);

  TestUndirectedGraph<> g5;
  g5 = std::move(g4);
  BOOST_CHECK_EQUAL(g5.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g5.num_edges(), 1);
}

BOOST_FIXTURE_TEST_CASE(test_graph_accessors, Fixture) {
  BOOST_CHECK(g.contains(v1));
  BOOST_CHECK(g.contains(v2));
  BOOST_CHECK(g.contains(v3));
  BOOST_CHECK(g.contains(v4));
  BOOST_CHECK(g.contains(e12));
  BOOST_CHECK(g.contains(e23));
  BOOST_CHECK(g.contains(e24));
  BOOST_CHECK(g.contains(g.root()));

  BOOST_CHECK_EQUAL(g.in_degree(v2), 3);
  BOOST_CHECK_EQUAL(g.out_degree(v2), 3);
  BOOST_CHECK_EQUAL(g.degree(v2), 3);

  std::vector<UndirectedGraph::Vertex*> neighbors;
  for (auto* vertex : g.adjacent_vertices(v2)) {
    neighbors.push_back(vertex);
  }
  std::vector<UndirectedGraph::Vertex*> expected_neighbors = {v1, v3, v4};
  BOOST_CHECK(neighbors == expected_neighbors);

  std::vector<UndirectedGraph::edge_descriptor> out;
  for (auto edge : g.out_edges(v2)) {
    out.push_back(edge);
  }
  std::vector<UndirectedGraph::edge_descriptor> expected_out = {
    UndirectedGraph::edge_descriptor(e12.reverse()),
    UndirectedGraph::edge_descriptor(e23),
    UndirectedGraph::edge_descriptor(e24)
  };
  BOOST_CHECK(out == expected_out);

  std::vector<UndirectedGraph::edge_descriptor> in;
  for (auto edge : g.in_edges(v2)) {
    in.push_back(edge);
  }
  std::vector<UndirectedGraph::edge_descriptor> expected_in = {
    UndirectedGraph::edge_descriptor(e12),
    UndirectedGraph::edge_descriptor(e23.reverse()),
    UndirectedGraph::edge_descriptor(e24.reverse())
  };
  BOOST_CHECK(in == expected_in);
}

BOOST_FIXTURE_TEST_CASE(test_structure_queries_and_traversals, Fixture) {
  BOOST_CHECK(g.is_connected());
  BOOST_CHECK(g.is_tree());

  std::vector<UndirectedGraph::edge_descriptor> preorder;
  g.pre_order_traversal(v1, [&](UndirectedGraph::edge_descriptor e) {
    preorder.push_back(e);
  });
  BOOST_CHECK_EQUAL(preorder.size(), g.num_edges());

  std::vector<UndirectedGraph::edge_descriptor> postorder;
  g.post_order_traversal(v1, [&](UndirectedGraph::edge_descriptor e) {
    postorder.push_back(e);
  });
  BOOST_CHECK_EQUAL(postorder.size(), g.num_edges());

  std::vector<UndirectedGraph::edge_descriptor> mpp;
  g.mpp_traversal(v1, [&](UndirectedGraph::edge_descriptor e) {
    mpp.push_back(e);
  });
  BOOST_CHECK_EQUAL(mpp.size(), g.num_edges() * 2);
}

BOOST_FIXTURE_TEST_CASE(test_remove_and_clear, Fixture) {
  BOOST_CHECK_EQUAL(g.remove_edge(v1, v3), 0);
  BOOST_CHECK_EQUAL(g.remove_edge(v1, v2), 1);
  BOOST_CHECK_EQUAL(g.num_edges(), 2);

  g.clear_vertex(v2);
  BOOST_CHECK_EQUAL(g.num_edges(), 0);
  BOOST_CHECK_EQUAL(g.degree(v1), 0);
  BOOST_CHECK_EQUAL(g.degree(v2), 0);
  BOOST_CHECK_EQUAL(g.degree(v3), 0);
  BOOST_CHECK_EQUAL(g.degree(v4), 0);

  g.remove_vertex(v1);
  BOOST_CHECK_EQUAL(g.num_vertices(), 3);

  g.clear();
  BOOST_CHECK(g.empty());
}

BOOST_AUTO_TEST_CASE(test_untyped_property_layout) {
  TestUndirectedGraph<> g;
  auto* v = g.add_vertex();
  auto* u = g.add_vertex();
  auto e = g.add_edge(v, u);

  BOOST_CHECK(g.property(v).type_info == typeid(void));
  BOOST_CHECK_EQUAL(g.property(v).ptr, nullptr);
  BOOST_CHECK(g.property(e).type_info == typeid(void));
  BOOST_CHECK_EQUAL(g.property(e).ptr, nullptr);
}

BOOST_AUTO_TEST_CASE(test_typed_property_handling) {
  CountingProperty::alive_count = 0;

  TestUndirectedGraph<CountingProperty, CountingProperty> g;
  auto* v1 = g.add_vertex();
  auto* v2 = g.add_vertex();
  auto e = g.add_edge(v1, v2);

  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 3);

  g.vertex_property(v1) = CountingProperty(10);
  g.vertex_property(v2) = CountingProperty(20);
  g.edge_property(e) = CountingProperty(30);

  BOOST_CHECK_EQUAL(g.vertex_property(v1).value, 10);
  BOOST_CHECK_EQUAL(g.vertex_property(v2).value, 20);
  BOOST_CHECK_EQUAL(g.edge_property(e).value, 30);

  g.clear();
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}
