#define BOOST_TEST_MODULE markov_network_property_layout
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/markov_network.hpp>

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

BOOST_AUTO_TEST_CASE(base_markov_network_uses_null_edge_property_pointer) {
  MarkovNetwork mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));

  auto [e, inserted] = mn.add_edge(a, b);
  BOOST_CHECK(inserted);
  BOOST_CHECK_EQUAL(mn.property(e).ptr, nullptr);
}

BOOST_AUTO_TEST_CASE(typed_vertex_and_edge_property_addresses_and_lifetime) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  MarkovNetworkT<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(mn.add_vertex(a, VertexProperty(10)));
  BOOST_CHECK(mn.add_vertex(b, VertexProperty(20)));
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(mn[a].value, 10);
  BOOST_CHECK_EQUAL(mn[b].value, 20);
  BOOST_CHECK_EQUAL(static_cast<void*>(&mn[a]), mn.property(a).ptr);

  auto [e, inserted] = mn.add_edge(a, b, EdgeProperty(30));
  BOOST_CHECK(inserted);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(mn[e].value, 30);
  BOOST_CHECK_EQUAL(static_cast<void*>(&mn[e]), mn.property(e).ptr);

  mn.remove_edge(a, b);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);

  mn.remove_vertex(a);
  mn.remove_vertex(b);
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(default_constructed_properties_and_clear) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  MarkovNetworkT<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));
  auto [e, inserted] = mn.add_edge(a, b);
  BOOST_CHECK(inserted);

  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(mn[a].value, 0);
  BOOST_CHECK_EQUAL(mn[b].value, 0);
  BOOST_CHECK_EQUAL(mn[e].value, 0);

  mn.clear();
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(add_edge_does_not_overwrite_existing_property) {
  VertexProperty::alive_count = 0;
  EdgeProperty::alive_count = 0;

  MarkovNetworkT<VertexProperty, EdgeProperty> mn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(mn.add_vertex(a));
  BOOST_CHECK(mn.add_vertex(b));

  auto [e1, inserted1] = mn.add_edge(a, b, EdgeProperty(1));
  BOOST_CHECK(inserted1);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(mn[e1].value, 1);

  auto [e2, inserted2] = mn.add_edge(a, b, EdgeProperty(9));
  BOOST_CHECK(!inserted2);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(mn[e2].value, 1);

  mn.clear();
  BOOST_CHECK_EQUAL(VertexProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(EdgeProperty::alive_count, 0);
}

} // namespace libgm
