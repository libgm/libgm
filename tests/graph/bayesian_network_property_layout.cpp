#define BOOST_TEST_MODULE bayesian_network_property_layout
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/bayesian_network.hpp>

namespace libgm {
namespace {

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

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(property_pointer_matches_typed_reference) {
  CountingProperty::alive_count = 0;

  BayesianNetworkT<CountingProperty> bn;
  Arg a = make_arg("a");

  bn.add_vertex(a, Domain(), CountingProperty(7));

  BOOST_CHECK_EQUAL(bn.num_vertices(), 1);
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(bn[a].value, 7);
  BOOST_CHECK_EQUAL(static_cast<void*>(&bn[a]), bn.property(a).ptr);

  bn.remove_vertex(a);
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(overwrite_reconstructs_property_without_leak) {
  CountingProperty::alive_count = 0;

  BayesianNetworkT<CountingProperty> bn;
  Arg a = make_arg("a");

  bn.add_vertex(a, Domain(), CountingProperty(1));
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(bn[a].value, 1);

  bn.add_vertex(a, Domain(), CountingProperty(9));
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(bn[a].value, 9);

  bn.clear();
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(default_constructed_property_and_clear) {
  CountingProperty::alive_count = 0;

  BayesianNetworkT<CountingProperty> bn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");

  bn.add_vertex(a, Domain());
  bn.add_vertex(b, Domain());

  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(bn[a].value, 0);
  BOOST_CHECK_EQUAL(bn[b].value, 0);

  bn.clear();
  BOOST_CHECK_EQUAL(CountingProperty::alive_count, 0);
}

} // namespace libgm
