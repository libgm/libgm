#define BOOST_TEST_MODULE factor_graph
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/factor_graph.hpp>

#include <unordered_set>
#include <vector>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

struct Fixture {
  Fixture() {
    a = make_arg("a");
    b = make_arg("b");
    c = make_arg("c");
    d = make_arg("d");
    e = make_arg("e");

    fg.add_argument(a);
    fg.add_argument(b);
    fg.add_argument(c);
    fg.add_argument(d);
    fg.add_argument(e);

    f_ab = fg.add_factor({a, b});
    f_bc = fg.add_factor({b, c});
    f_cde = fg.add_factor({c, d, e});
  }

  Arg a;
  Arg b;
  Arg c;
  Arg d;
  Arg e;

  FactorGraph fg;
  FactorGraph::Factor* f_ab = nullptr;
  FactorGraph::Factor* f_bc = nullptr;
  FactorGraph::Factor* f_cde = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_copy_move) {
  FactorGraph g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK_EQUAL(g1.num_arguments(), 0);
  BOOST_CHECK_EQUAL(g1.num_factors(), 0);

  Arg a = make_arg("ca");
  Arg b = make_arg("cb");
  BOOST_CHECK(g1.add_argument(a));
  BOOST_CHECK(g1.add_argument(b));
  FactorGraph::Factor* f = g1.add_factor({a, b});

  BOOST_CHECK(!g1.empty());
  BOOST_CHECK(g1.contains(a));
  BOOST_CHECK(g1.contains(b));
  BOOST_CHECK(g1.contains(f));
  BOOST_CHECK(g1.contains(a, f));
  BOOST_CHECK(g1.contains(b, f));
  BOOST_CHECK_EQUAL(g1.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g1.num_factors(), 1);

  FactorGraph g2(g1);
  BOOST_CHECK_EQUAL(g2.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g2.num_factors(), 1);

  FactorGraph g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g3.num_factors(), 1);

  FactorGraph g4(std::move(g3));
  BOOST_CHECK_EQUAL(g4.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g4.num_factors(), 1);

  FactorGraph g5;
  g5 = std::move(g4);
  BOOST_CHECK_EQUAL(g5.num_arguments(), 2);
  BOOST_CHECK_EQUAL(g5.num_factors(), 1);
}

BOOST_FIXTURE_TEST_CASE(test_accessors_contains_and_degree, Fixture) {
  BOOST_CHECK(fg.contains(a));
  BOOST_CHECK(fg.contains(b));
  BOOST_CHECK(fg.contains(c));
  BOOST_CHECK(fg.contains(d));
  BOOST_CHECK(fg.contains(e));
  BOOST_CHECK(fg.contains(f_ab));
  BOOST_CHECK(fg.contains(f_bc));
  BOOST_CHECK(fg.contains(f_cde));

  BOOST_CHECK_EQUAL(fg.num_arguments(), 5);
  BOOST_CHECK_EQUAL(fg.num_factors(), 3);

  BOOST_CHECK_EQUAL(fg.degree(a), 1);
  BOOST_CHECK_EQUAL(fg.degree(b), 2);
  BOOST_CHECK_EQUAL(fg.degree(c), 2);
  BOOST_CHECK_EQUAL(fg.degree(d), 1);
  BOOST_CHECK_EQUAL(fg.degree(e), 1);

  BOOST_CHECK_EQUAL(fg.degree(f_ab), 2);
  BOOST_CHECK_EQUAL(fg.degree(f_bc), 2);
  BOOST_CHECK_EQUAL(fg.degree(f_cde), 3);

  BOOST_CHECK(fg.arguments(f_ab) == Domain({a, b}));
  BOOST_CHECK(fg.arguments(f_bc) == Domain({b, c}));
  BOOST_CHECK(fg.arguments(f_cde) == Domain({c, d, e}));

  std::unordered_set<FactorGraph::Factor*> factors_of_c = {f_bc, f_cde};
  for (FactorGraph::Factor* f : fg.factors(c)) {
    BOOST_CHECK_EQUAL(factors_of_c.erase(f), 1);
  }
  BOOST_CHECK(factors_of_c.empty());

  std::unordered_set<Arg> all_args = {a, b, c, d, e};
  for (Arg u : fg.arguments()) {
    BOOST_CHECK_EQUAL(all_args.erase(u), 1);
  }
  BOOST_CHECK(all_args.empty());

  size_t factor_count = 0;
  for (FactorGraph::Factor* f : fg.factors()) {
    BOOST_CHECK(fg.contains(f));
    ++factor_count;
  }
  BOOST_CHECK_EQUAL(factor_count, fg.num_factors());
}

BOOST_FIXTURE_TEST_CASE(test_argument_factor_incidences, Fixture) {
  std::unordered_set<FactorGraph::Factor*> factors_of_a = {f_ab};
  for (FactorGraph::Factor* f : fg.factors(a)) {
    BOOST_CHECK_EQUAL(factors_of_a.erase(f), 1);
  }
  BOOST_CHECK(factors_of_a.empty());

  std::unordered_set<FactorGraph::Factor*> factors_of_b = {f_ab, f_bc};
  for (FactorGraph::Factor* f : fg.factors(b)) {
    BOOST_CHECK_EQUAL(factors_of_b.erase(f), 1);
  }
  BOOST_CHECK(factors_of_b.empty());

  std::unordered_set<FactorGraph::Factor*> factors_of_c = {f_bc, f_cde};
  for (FactorGraph::Factor* f : fg.factors(c)) {
    BOOST_CHECK_EQUAL(factors_of_c.erase(f), 1);
  }
  BOOST_CHECK(factors_of_c.empty());

  std::unordered_set<FactorGraph::Factor*> factors_of_d = {f_cde};
  for (FactorGraph::Factor* f : fg.factors(d)) {
    BOOST_CHECK_EQUAL(factors_of_d.erase(f), 1);
  }
  BOOST_CHECK(factors_of_d.empty());

  std::unordered_set<FactorGraph::Factor*> factors_of_e = {f_cde};
  for (FactorGraph::Factor* f : fg.factors(e)) {
    BOOST_CHECK_EQUAL(factors_of_e.erase(f), 1);
  }
  BOOST_CHECK(factors_of_e.empty());

  BOOST_CHECK(fg.arguments(f_ab) == Domain({a, b}));
  BOOST_CHECK(fg.arguments(f_bc) == Domain({b, c}));
  BOOST_CHECK(fg.arguments(f_cde) == Domain({c, d, e}));
}

BOOST_FIXTURE_TEST_CASE(test_markov_network, Fixture) {
  MarkovNetwork mn = fg.markov_network();

  BOOST_CHECK_EQUAL(mn.num_vertices(), 5);
  BOOST_CHECK(mn.contains(a, b));
  BOOST_CHECK(mn.contains(b, c));
  BOOST_CHECK(mn.contains(c, d));
  BOOST_CHECK(mn.contains(c, e));
  BOOST_CHECK(mn.contains(d, e));

  BOOST_CHECK(!mn.contains(a, c));
  BOOST_CHECK(!mn.contains(a, e));
  BOOST_CHECK(!mn.contains(b, e));
}

BOOST_FIXTURE_TEST_CASE(test_updates_and_removals, Fixture) {
  auto expect_factors = [&](Arg u, std::initializer_list<FactorGraph::Factor*> expected) {
    std::unordered_set<FactorGraph::Factor*> remaining(expected.begin(), expected.end());
    for (FactorGraph::Factor* f : fg.factors(u)) {
      BOOST_CHECK_EQUAL(remaining.erase(f), 1);
    }
    BOOST_CHECK(remaining.empty());
  };

  BOOST_CHECK(!fg.add_argument(a));
  BOOST_CHECK_EQUAL(fg.num_arguments(), 5);

  expect_factors(b, {f_ab, f_bc});
  expect_factors(c, {f_bc, f_cde});
  fg.remove_factor(f_bc);
  BOOST_CHECK_EQUAL(fg.num_factors(), 2);
  BOOST_CHECK_EQUAL(fg.degree(b), 1);
  BOOST_CHECK_EQUAL(fg.degree(c), 1);
  expect_factors(b, {f_ab});
  expect_factors(c, {f_cde});

  expect_factors(a, {f_ab});
  expect_factors(b, {f_ab});
  fg.remove_factor(f_ab);
  BOOST_CHECK_EQUAL(fg.num_factors(), 1);
  BOOST_CHECK_EQUAL(fg.degree(a), 0);
  BOOST_CHECK_EQUAL(fg.degree(b), 0);
  expect_factors(a, {});
  expect_factors(b, {});
  expect_factors(c, {f_cde});

  fg.remove_argument(a);
  fg.remove_argument(b);
  BOOST_CHECK_EQUAL(fg.num_arguments(), 3);
  BOOST_CHECK(!fg.contains(a));
  BOOST_CHECK(!fg.contains(b));

  expect_factors(c, {f_cde});
  expect_factors(d, {f_cde});
  expect_factors(e, {f_cde});
  fg.remove_factor(f_cde);
  expect_factors(c, {});
  expect_factors(d, {});
  expect_factors(e, {});
  fg.remove_argument(c);
  fg.remove_argument(d);
  fg.remove_argument(e);
  BOOST_CHECK(fg.empty());
  BOOST_CHECK_EQUAL(fg.num_arguments(), 0);
  BOOST_CHECK_EQUAL(fg.num_factors(), 0);

  fg.clear();
  BOOST_CHECK(fg.empty());
}
