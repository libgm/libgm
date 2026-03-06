#define BOOST_TEST_MODULE cluster_graph_typed
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/cluster_graph.hpp>

namespace libgm {
namespace {

struct VertexProperty {
  static int alive_count;
  int value = 0;

  explicit VertexProperty(int v = 0)
    : value(v) {
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

  explicit EdgeProperty(int v = 0)
    : value(v) {
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

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(vertex_and_edge_property_addresses_and_lifetime) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  ClusterGraphT<VertexProperty, EdgeProperty> cg;

  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");

  ClusterGraph::Vertex* v1 = cg.add_vertex({a, b}, VertexProperty(10));
  ClusterGraph::Vertex* v2 = cg.add_vertex({b, c}, VertexProperty(20));
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(cg[v1].value, 10);
  BOOST_CHECK_EQUAL(cg[v2].value, 20);
  BOOST_CHECK_EQUAL(static_cast<void*>(&cg[v1]), cg.property(v1).ptr);

  auto e = cg.add_edge(v1, v2, {b}, EdgeProperty(30));
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(cg[e].value, 30);
  BOOST_CHECK_EQUAL(static_cast<void*>(&cg[e]), cg.property(e).ptr);

  cg.remove_edge(e);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);

  cg.remove_vertex(v1);
  cg.remove_vertex(v2);
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(default_constructed_properties_and_clear) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  ClusterGraphT<VertexProperty, EdgeProperty> cg;

  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");

  ClusterGraph::Vertex* v1 = cg.add_vertex({a, b});
  ClusterGraph::Vertex* v2 = cg.add_vertex({b, c});
  auto e = cg.add_edge(v1, v2, {b});

  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(cg[v1].value, 0);
  BOOST_CHECK_EQUAL(cg[v2].value, 0);
  BOOST_CHECK_EQUAL(cg[e].value, 0);

  cg.clear();
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(assignment_overwrites_without_leaking_instances) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  ClusterGraphT<VertexProperty, EdgeProperty> cg;

  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");

  ClusterGraph::Vertex* v1 = cg.add_vertex({a, b});
  ClusterGraph::Vertex* v2 = cg.add_vertex({b, c});
  auto e = cg.add_edge(v1, v2, {b});

  cg[v1] = VertexProperty(77);
  cg[e] = EdgeProperty(88);

  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(cg[v1].value, 77);
  BOOST_CHECK_EQUAL(cg[e].value, 88);

  cg.clear();
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);
}

} // namespace libgm
