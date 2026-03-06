#define BOOST_TEST_MODULE factor_graph_property_layout
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/factor_graph.hpp>

namespace libgm {
namespace {

struct ArgumentProperty {
  static int alive_count;
  int value = 0;

  explicit ArgumentProperty(int v = 0)
    : value(v) {
    ++alive_count;
  }

  ArgumentProperty(const ArgumentProperty& other)
    : value(other.value) {
    ++alive_count;
  }

  ArgumentProperty(ArgumentProperty&& other) noexcept
    : value(other.value) {
    ++alive_count;
  }

  ArgumentProperty& operator=(const ArgumentProperty&) = default;
  ArgumentProperty& operator=(ArgumentProperty&&) noexcept = default;

  ~ArgumentProperty() {
    --alive_count;
  }
};

int ArgumentProperty::alive_count = 0;

struct FactorProperty {
  static int alive_count;
  int value = 0;

  explicit FactorProperty(int v = 0)
    : value(v) {
    ++alive_count;
  }

  FactorProperty(const FactorProperty& other)
    : value(other.value) {
    ++alive_count;
  }

  FactorProperty(FactorProperty&& other) noexcept
    : value(other.value) {
    ++alive_count;
  }

  FactorProperty& operator=(const FactorProperty&) = default;
  FactorProperty& operator=(FactorProperty&&) noexcept = default;

  ~FactorProperty() {
    --alive_count;
  }
};

int FactorProperty::alive_count = 0;

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(argument_and_factor_property_addresses_and_lifetime) {
  ArgumentProperty::alive_count = 0;
  FactorProperty::alive_count = 0;

  FactorGraphT<ArgumentProperty, FactorProperty> fg;

  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(fg.add_argument(a, ArgumentProperty(10)));
  BOOST_CHECK(fg.add_argument(b, ArgumentProperty(20)));
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(fg[a].value, 10);
  BOOST_CHECK_EQUAL(fg[b].value, 20);
  BOOST_CHECK_EQUAL(static_cast<void*>(&fg[a]), fg.property(a).ptr);

  FactorGraph::Factor* f = fg.add_factor({a, b}, FactorProperty(30));
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(fg[f].value, 30);
  BOOST_CHECK_EQUAL(static_cast<void*>(&fg[f]), fg.property(f).ptr);

  fg.remove_factor(f);
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 0);

  fg.remove_argument(a);
  fg.remove_argument(b);
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(default_constructed_properties_and_clear) {
  ArgumentProperty::alive_count = 0;
  FactorProperty::alive_count = 0;

  FactorGraphT<ArgumentProperty, FactorProperty> fg;

  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(fg.add_argument(a));
  BOOST_CHECK(fg.add_argument(b));
  FactorGraph::Factor* f = fg.add_factor({a, b});

  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(fg[a].value, 0);
  BOOST_CHECK_EQUAL(fg[b].value, 0);
  BOOST_CHECK_EQUAL(fg[f].value, 0);

  fg.clear();
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 0);
}

BOOST_AUTO_TEST_CASE(add_argument_does_not_overwrite_existing_property) {
  ArgumentProperty::alive_count = 0;
  FactorProperty::alive_count = 0;

  FactorGraphT<ArgumentProperty, FactorProperty> fg;
  Arg a = make_arg("a");

  BOOST_CHECK(fg.add_argument(a, ArgumentProperty(1)));
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(fg[a].value, 1);

  BOOST_CHECK(!fg.add_argument(a, ArgumentProperty(9)));
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(fg[a].value, 1);

  fg.clear();
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 0);
}

} // namespace libgm
