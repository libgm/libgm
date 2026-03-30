#define BOOST_TEST_MODULE factor_graph
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/model/factor_graph.hpp>

#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

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

struct Fixture {
  Fixture() {
    a = make_arg("a");
    b = make_arg("b");
    c = make_arg("c");
    d = make_arg("d");
    e = make_arg("e");

    va = fg.add_argument(a);
    vb = fg.add_argument(b);
    vc = fg.add_argument(c);
    vd = fg.add_argument(d);
    ve = fg.add_argument(e);

    f_ab = fg.add_factor({a, b});
    f_bc = fg.add_factor({b, c});
    f_cde = fg.add_factor({c, d, e});
  }

  Arg a;
  Arg b;
  Arg c;
  Arg d;
  Arg e;
  FactorGraph<>::Argument* va = nullptr;
  FactorGraph<>::Argument* vb = nullptr;
  FactorGraph<>::Argument* vc = nullptr;
  FactorGraph<>::Argument* vd = nullptr;
  FactorGraph<>::Argument* ve = nullptr;

  FactorGraph<> fg;
  FactorGraph<>::Factor* f_ab = nullptr;
  FactorGraph<>::Factor* f_bc = nullptr;
  FactorGraph<>::Factor* f_cde = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_edge_descriptor_basics) {
  using lr_edge_type = FactorGraph<>::edge12_descriptor;
  using rl_edge_type = FactorGraph<>::edge21_descriptor;

  Arg a = make_arg("edge_a");
  Arg b = make_arg("edge_b");
  FactorGraph<> fg;
  auto* va = fg.add_argument(a);
  auto* vb = fg.add_argument(b);
  auto* f = fg.add_factor({a, b});

  lr_edge_type lr(va, f);
  BOOST_CHECK_EQUAL(lr.source(), va);
  BOOST_CHECK_EQUAL(lr.target(), f);

  rl_edge_type rl(f, vb);
  BOOST_CHECK_EQUAL(rl.source(), f);
  BOOST_CHECK_EQUAL(rl.target(), vb);
}

BOOST_AUTO_TEST_CASE(test_constructors_and_copy_move) {
  FactorGraph<> g1;
  BOOST_CHECK(g1.empty());
  Arg a = make_arg("ca");
  Arg b = make_arg("cb");
  auto* va = g1.add_argument(a);
  auto* vb = g1.add_argument(b);
  auto* f = g1.add_factor({a, b});

  BOOST_CHECK(g1.contains(a));
  BOOST_CHECK_EQUAL(g1.vertex(a), va);
  BOOST_CHECK_EQUAL(g1.vertex(b), vb);
  BOOST_CHECK(g1.contains(f));
  BOOST_CHECK(g1.contains(a, f));
  BOOST_CHECK_EQUAL(g1.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g1.num_factors(), 1);

  FactorGraph<> g2(g1);
  BOOST_CHECK_EQUAL(g2.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g2.num_factors(), 1);
}

BOOST_AUTO_TEST_CASE(base_factor_graph_uses_null_property_pointers) {
  FactorGraph<> fg;
  Arg a = make_arg("prop_a");
  Arg b = make_arg("prop_b");
  auto* va = fg.add_argument(a);
  fg.add_argument(b);
  auto* f = fg.add_factor({a, b});

  BOOST_CHECK(fg.property(va).type_info == typeid(Annotated<Arg, void>));
  BOOST_CHECK(fg.property(va).ptr != nullptr);
  BOOST_CHECK(fg.property(f).type_info == typeid(Annotated<Domain, void>));
  BOOST_CHECK(fg.property(f).ptr != nullptr);
}

BOOST_FIXTURE_TEST_CASE(test_accessors_contains_and_degree, Fixture) {
  BOOST_CHECK_EQUAL(fg.num_arguments(), 5);
  BOOST_CHECK_EQUAL(fg.num_factors(), 3);
  BOOST_CHECK_EQUAL(fg.degree(a), 1);
  BOOST_CHECK_EQUAL(fg.degree(b), 2);
  BOOST_CHECK_EQUAL(fg.degree(f_ab), 2);
  BOOST_CHECK_EQUAL(fg.arguments(f_cde), Domain({c, d, e}));
  BOOST_CHECK(std::ranges::equal(fg.factors(c), std::vector<FactorGraph<>::Factor*>{f_bc, f_cde}));
}

BOOST_FIXTURE_TEST_CASE(test_in_and_out_edges, Fixture) {
  std::vector<FactorGraph<>::edge12_descriptor> expected_out_b = {{vb, f_ab}, {vb, f_bc}};
  BOOST_CHECK(std::ranges::equal(fg.out_edges(vb), expected_out_b));

  std::vector<FactorGraph<>::edge21_descriptor> expected_in_b = {
    {f_ab, vb},
    {f_bc, vb}
  };
  BOOST_CHECK(std::ranges::equal(fg.in_edges(vb), expected_in_b));
}

BOOST_FIXTURE_TEST_CASE(test_markov_network, Fixture) {
  MarkovNetwork mn = fg.markov_network();
  BOOST_CHECK_EQUAL(mn.num_vertices(), 5);
  BOOST_CHECK(mn.contains(a, b));
  BOOST_CHECK(mn.contains(b, c));
  BOOST_CHECK(mn.contains(c, d));
}

BOOST_FIXTURE_TEST_CASE(test_updates_and_removals, Fixture) {
  BOOST_CHECK_THROW(fg.add_argument(a), std::invalid_argument);
  fg.remove_factor(f_bc);
  BOOST_CHECK_EQUAL(fg.num_factors(), 2);
  fg.remove_argument(a);
  BOOST_CHECK(!fg.contains(a));
  fg.clear();
  BOOST_CHECK(fg.empty());
}

BOOST_AUTO_TEST_CASE(argument_and_factor_property_addresses_and_lifetime) {
  ArgumentProperty::alive_count = 0;
  FactorProperty::alive_count = 0;

  FactorGraph<ArgumentProperty, FactorProperty> fg;
  Arg a = make_arg("typed_a");
  Arg b = make_arg("typed_b");

  auto* va = fg.add_argument(a, ArgumentProperty(10));
  fg.add_argument(b, ArgumentProperty(20));
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 2);
  BOOST_CHECK_EQUAL(fg[a].value, 10);
  BOOST_CHECK_EQUAL(static_cast<void*>(&fg[a]), fg.property(va).ptr);

  auto* f = fg.add_factor({a, b}, FactorProperty(30));
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 1);
  BOOST_CHECK_EQUAL(fg[f].value, 30);
  BOOST_CHECK_EQUAL(static_cast<void*>(&fg[f]), fg.property(f).ptr);

  fg.clear();
  BOOST_CHECK_EQUAL(ArgumentProperty::alive_count, 0);
  BOOST_CHECK_EQUAL(FactorProperty::alive_count, 0);
}
