#define BOOST_TEST_MODULE bayesian_network
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/model/bayesian_network.hpp>

#include <set>
#include <stdexcept>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

struct Fixture {
  Fixture() {
    x0 = make_arg("x0");
    x1 = make_arg("x1");
    x2 = make_arg("x2");
    x3 = make_arg("x3");
    x4 = make_arg("x4");

    v0 = bn.add_vertex(x0, {});
    v1 = bn.add_vertex(x1, {});
    v2 = bn.add_vertex(x2, {x1});
    v3 = bn.add_vertex(x3, {x1, x2});
    v4 = bn.add_vertex(x4, {x0, x3});
  }

  BayesianNetwork<> bn;
  Arg x0;
  Arg x1;
  Arg x2;
  Arg x3;
  Arg x4;
  BayesianNetwork<>::Vertex* v0 = nullptr;
  BayesianNetwork<>::Vertex* v1 = nullptr;
  BayesianNetwork<>::Vertex* v2 = nullptr;
  BayesianNetwork<>::Vertex* v3 = nullptr;
  BayesianNetwork<>::Vertex* v4 = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_empty) {
  BayesianNetwork<> bn;
  BOOST_CHECK(bn.empty());
  BOOST_CHECK_EQUAL(bn.num_vertices(), 0);
  BOOST_CHECK_EQUAL(bn.num_edges(), 0);

  Arg a = make_arg("a");
  Arg b = make_arg("b");

  auto* va = bn.add_vertex(a, {});
  auto* vb = bn.add_vertex(b, {a});
  BOOST_CHECK_EQUAL(bn.vertex(a), va);
  BOOST_CHECK_EQUAL(bn.vertex(b), vb);

  BayesianNetwork<> copy(bn);
  BOOST_CHECK(copy.contains(a));
  BOOST_CHECK(copy.contains(copy.vertex(a), copy.vertex(b)));

  BayesianNetwork<> assigned;
  assigned = bn;
  BOOST_CHECK(assigned.contains(a));
  BOOST_CHECK(assigned.contains(assigned.vertex(a), assigned.vertex(b)));
}

BOOST_FIXTURE_TEST_CASE(test_argument_mapping, Fixture) {
  BOOST_CHECK_EQUAL(bn.vertex(x0), v0);
  BOOST_CHECK_EQUAL(bn.vertex(x1), v1);
  BOOST_CHECK_EQUAL(bn.argument(v2), x2);
  BOOST_CHECK_EQUAL(bn.argument(v3), x3);

  std::set<Arg> seen;
  for (auto* v : bn.vertices()) {
    seen.insert(bn.argument(v));
  }
  BOOST_CHECK_EQUAL(seen.size(), 5);
  BOOST_CHECK(seen.count(x0));
  BOOST_CHECK(seen.count(x4));
}

BOOST_AUTO_TEST_CASE(test_add_vertex_parent_validation) {
  BayesianNetwork<> bn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg z = make_arg("z");

  bn.add_vertex(a, {});
  bn.add_vertex(b, {});

  BOOST_CHECK_THROW(bn.add_vertex(c, {c}), std::invalid_argument);
  BOOST_CHECK_THROW(bn.add_vertex(c, {z}), std::out_of_range);
  BOOST_CHECK_THROW(bn.add_vertex(b, {a}), std::invalid_argument);
}

BOOST_FIXTURE_TEST_CASE(test_set_parents_preserves_argument_mapping, Fixture) {
  bn.set_parents(x3, {x0});
  BOOST_CHECK_EQUAL(bn.vertex(x3), v3);
  BOOST_CHECK_EQUAL(bn.argument(v3), x3);
  BOOST_CHECK_EQUAL(bn.in_degree(v3), 1);
  BOOST_CHECK(bn.contains(v0, v3));
  BOOST_CHECK(!bn.contains(v1, v3));
  BOOST_CHECK(!bn.contains(v2, v3));
}

BOOST_FIXTURE_TEST_CASE(test_set_parents_validation, Fixture) {
  Arg z = make_arg("z");
  BOOST_CHECK_THROW(bn.set_parents(x3, {x3}), std::invalid_argument);
  BOOST_CHECK_THROW(bn.set_parents(x3, {z}), std::out_of_range);
}

BOOST_FIXTURE_TEST_CASE(test_remove_and_clear_update_mapping, Fixture) {
  BOOST_CHECK_EQUAL(bn.remove_vertex(x4), 1);
  BOOST_CHECK(!bn.contains(x4));
  BOOST_CHECK_EQUAL(bn.num_vertices(), 4);

  bn.remove_in_edges(x3);
  BOOST_CHECK(bn.parents(v3).empty());
  BOOST_CHECK_EQUAL(bn.in_degree(v3), 0);

  bn.clear();
  BOOST_CHECK(bn.empty());
  BOOST_CHECK(!bn.contains(x0));
}

BOOST_FIXTURE_TEST_CASE(test_markov_network_uses_argument_mapping, Fixture) {
  MarkovNetwork mn = bn.markov_network();

  BOOST_CHECK_EQUAL(mn.num_vertices(), 5);
  BOOST_CHECK_EQUAL(mn.num_edges(), 6);
  BOOST_CHECK(mn.contains(x1, x2));
  BOOST_CHECK(mn.contains(x1, x3));
  BOOST_CHECK(mn.contains(x2, x3));
  BOOST_CHECK(mn.contains(x0, x3));
  BOOST_CHECK(mn.contains(x0, x4));
  BOOST_CHECK(mn.contains(x3, x4));
}
