#define BOOST_TEST_MODULE bipartite_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/bipartite_graph.hpp>

#include <ranges>
#include <utility>
#include <vector>

using namespace libgm;

namespace {

template <typename VP1 = void, typename VP2 = void>
struct TestBipartiteGraph : BipartiteGraph {
  TestBipartiteGraph()
    : BipartiteGraph(property_layout<VP1>(), property_layout<VP2>()) {}

  using BipartiteGraph::add_vertex1;
  using BipartiteGraph::add_vertex2;
  using BipartiteGraph::property;

  using vertex1_property_reference = std::add_lvalue_reference_t<VP1>;
  using const_vertex1_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP1>>;
  using vertex2_property_reference = std::add_lvalue_reference_t<VP2>;
  using const_vertex2_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP2>>;

  vertex1_property_reference operator[](Vertex1* u) {
    if constexpr (std::is_void_v<VP1>) {
      return;
    } else {
      return opaque_cast<VP1>(property(u));
    }
  }

  const_vertex1_property_reference operator[](Vertex1* u) const {
    if constexpr (std::is_void_v<VP1>) {
      return;
    } else {
      return opaque_cast<VP1>(property(u));
    }
  }

  vertex2_property_reference operator[](Vertex2* u) {
    if constexpr (std::is_void_v<VP2>) {
      return;
    } else {
      return opaque_cast<VP2>(property(u));
    }
  }

  const_vertex2_property_reference operator[](Vertex2* u) const {
    if constexpr (std::is_void_v<VP2>) {
      return;
    } else {
      return opaque_cast<VP2>(property(u));
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

  CountingProperty& operator=(const CountingProperty&) = default;
  CountingProperty& operator=(CountingProperty&&) noexcept = default;

  ~CountingProperty() {
    --alive_count;
  }
};

int CountingProperty::alive_count = 0;

struct Fixture {
  Fixture() {
    a = g.add_vertex1();
    b = g.add_vertex1();
    c = g.add_vertex1();
    d = g.add_vertex1();
    e = g.add_vertex1();

    f_ab = g.add_vertex2({a, b});
    f_bc = g.add_vertex2({b, c});
    f_cde = g.add_vertex2({c, d, e});
  }

  TestBipartiteGraph<> g;
  BipartiteGraph::Vertex1* a = nullptr;
  BipartiteGraph::Vertex1* b = nullptr;
  BipartiteGraph::Vertex1* c = nullptr;
  BipartiteGraph::Vertex1* d = nullptr;
  BipartiteGraph::Vertex1* e = nullptr;
  BipartiteGraph::Vertex2* f_ab = nullptr;
  BipartiteGraph::Vertex2* f_bc = nullptr;
  BipartiteGraph::Vertex2* f_cde = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_edge_descriptor_basics) {
  using edge12_type = BipartiteGraph::edge12_descriptor;
  using edge21_type = BipartiteGraph::edge21_descriptor;

  TestBipartiteGraph<> g;
  auto* a = g.add_vertex1();
  auto* b = g.add_vertex1();
  auto* f = g.add_vertex2({a, b});

  edge12_type lr(a, f);
  BOOST_CHECK(lr);
  BOOST_CHECK_EQUAL(lr.source(), a);
  BOOST_CHECK_EQUAL(lr.target(), f);

  edge21_type rl(f, b);
  BOOST_CHECK(rl);
  BOOST_CHECK_EQUAL(rl.source(), f);
  BOOST_CHECK_EQUAL(rl.target(), b);
}

BOOST_AUTO_TEST_CASE(test_constructors_and_copy_move) {
  TestBipartiteGraph<> g1;
  BOOST_CHECK(g1.empty());
  auto* a = g1.add_vertex1();
  auto* b = g1.add_vertex1();
  auto* f = g1.add_vertex2({a, b});

  BOOST_CHECK(g1.contains(a));
  BOOST_CHECK(g1.contains(b));
  BOOST_CHECK(g1.contains(f));
  BOOST_CHECK(g1.contains(a, f));
  BOOST_CHECK_EQUAL(g1.num_vertices1(), 2);
  BOOST_CHECK_EQUAL(g1.num_vertices2(), 1);

  TestBipartiteGraph<> g2(g1);
  BOOST_CHECK_EQUAL(g2.num_vertices1(), 2);
  BOOST_CHECK_EQUAL(g2.num_vertices2(), 1);

  TestBipartiteGraph<> g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_vertices1(), 2);
  BOOST_CHECK_EQUAL(g3.num_vertices2(), 1);
}

BOOST_AUTO_TEST_CASE(base_bipartite_graph_uses_null_property_pointers) {
  TestBipartiteGraph<> g;
  auto* a = g.add_vertex1();
  auto* f = g.add_vertex2({a});

  BOOST_CHECK(g.property(a).type_info == typeid(void));
  BOOST_CHECK_EQUAL(g.property(a).ptr, nullptr);
  BOOST_CHECK(g.property(f).type_info == typeid(void));
  BOOST_CHECK_EQUAL(g.property(f).ptr, nullptr);
}

BOOST_FIXTURE_TEST_CASE(test_accessors_contains_and_degree, Fixture) {
  BOOST_CHECK_EQUAL(g.num_vertices1(), 5);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 3);
  BOOST_CHECK_EQUAL(g.degree(a), 1);
  BOOST_CHECK_EQUAL(g.degree(b), 2);
  BOOST_CHECK_EQUAL(g.degree(c), 2);
  BOOST_CHECK_EQUAL(g.degree(f_ab), 2);
  BOOST_CHECK_EQUAL(g.degree(f_cde), 3);

  BOOST_CHECK(std::ranges::equal(g.neighbors(c), std::vector{f_bc, f_cde}));
  BOOST_CHECK(std::ranges::equal(g.neighbors(f_cde), std::vector{c, d, e}));
}

BOOST_FIXTURE_TEST_CASE(test_in_and_out_edges, Fixture) {
  std::vector<BipartiteGraph::edge12_descriptor> expected_out_b = {{b, f_ab}, {b, f_bc}};
  BOOST_CHECK(std::ranges::equal(g.out_edges(b), expected_out_b));

  std::vector<BipartiteGraph::edge21_descriptor> expected_in_b = {{f_ab, b}, {f_bc, b}};
  BOOST_CHECK(std::ranges::equal(g.in_edges(b), expected_in_b));

  std::vector<BipartiteGraph::edge21_descriptor> expected_out_cde = {{f_cde, c}, {f_cde, d}, {f_cde, e}};
  BOOST_CHECK(std::ranges::equal(g.out_edges(f_cde), expected_out_cde));

  std::vector<BipartiteGraph::edge12_descriptor> expected_in_cde = {{c, f_cde}, {d, f_cde}, {e, f_cde}};
  BOOST_CHECK(std::ranges::equal(g.in_edges(f_cde), expected_in_cde));
}

BOOST_FIXTURE_TEST_CASE(test_updates_and_removals, Fixture) {
  g.remove_vertex2(f_bc);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 2);
  BOOST_CHECK_EQUAL(g.degree(b), 1);
  BOOST_CHECK_EQUAL(g.degree(c), 1);

  g.remove_vertex1(a);
  BOOST_CHECK_EQUAL(g.num_vertices1(), 4);
  BOOST_CHECK_EQUAL(g.num_vertices2(), 1);
  BOOST_CHECK_EQUAL(g.degree(b), 0);

  g.clear();
  BOOST_CHECK(g.empty());
}

BOOST_AUTO_TEST_CASE(test_typed_property_addresses_and_lifetime) {
  CountingProperty::alive_count = 0;

  TestBipartiteGraph<CountingProperty, CountingProperty> g;
  auto* a = g.add_vertex1();
  auto* b = g.add_vertex1();
  auto* f = g.add_vertex2({a, b});

  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 3);
  g[a] = CountingProperty(10);
  g[f] = CountingProperty(20);
  BOOST_CHECK_EQUAL(g[a].value, 10);
  BOOST_CHECK_EQUAL(g[f].value, 20);
  BOOST_CHECK_EQUAL(static_cast<void*>(&g[a]), g.property(a).ptr);
  BOOST_CHECK_EQUAL(static_cast<void*>(&g[f]), g.property(f).ptr);

  g.clear();
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}
