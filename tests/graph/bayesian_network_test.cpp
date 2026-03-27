#define BOOST_TEST_MODULE bayesian_network
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/bayesian_network.hpp>

#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../predicates.hpp"

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

    // Structure:
    // x0, x1 (no parents)
    // x1 -> x2
    // x1, x2 -> x3
    // x0, x3 -> x4
    BOOST_CHECK(bn.add_vertex(x0, {}));
    BOOST_CHECK(bn.add_vertex(x1, {}));
    BOOST_CHECK(bn.add_vertex(x2, {x1}));
    BOOST_CHECK(bn.add_vertex(x3, {x1, x2}));
    BOOST_CHECK(bn.add_vertex(x4, {x0, x3}));
  }

  Arg x0;
  Arg x1;
  Arg x2;
  Arg x3;
  Arg x4;
  BayesianNetwork bn;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_empty) {
  BayesianNetwork g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK_EQUAL(g1.num_vertices(), 0);
  BOOST_CHECK_EQUAL(g1.num_edges(), 0);
  BOOST_CHECK(g1.vertices().empty());

  Arg a = make_arg("a");
  Arg b = make_arg("b");

  BOOST_CHECK(g1.add_vertex(a, {}));
  BOOST_CHECK(g1.add_vertex(b, {a}));

  BayesianNetwork g2(g1);
  BOOST_CHECK(!g2.empty());
  BOOST_CHECK_EQUAL(g2.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g2.num_edges(), 1);
  BOOST_CHECK(g2.contains(a));
  BOOST_CHECK(g2.contains(b));
  BOOST_CHECK(g2.contains(a, b));

  BayesianNetwork g3;
  g3 = g1;
  BOOST_CHECK_EQUAL(g3.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g3.num_edges(), 1);
  BOOST_CHECK(g3.contains(a, b));

  BayesianNetwork g4(std::move(g3));
  BOOST_CHECK_EQUAL(g4.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g4.num_edges(), 1);
  BOOST_CHECK(g4.contains(a, b));

  BayesianNetwork g5;
  g5 = std::move(g4);
  BOOST_CHECK_EQUAL(g5.num_vertices(), 2);
  BOOST_CHECK_EQUAL(g5.num_edges(), 1);
  BOOST_CHECK(g5.contains(a, b));
}

BOOST_AUTO_TEST_CASE(test_vertices_and_overwrite) {
  BayesianNetwork bn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");

  BOOST_CHECK(bn.add_vertex(a, {}));
  BOOST_CHECK(bn.add_vertex(b, {}));
  BOOST_CHECK(bn.add_vertex(c, {a, b}));

  BOOST_CHECK_EQUAL(bn.num_vertices(), 3);
  BOOST_CHECK_EQUAL(bn.num_edges(), 2);
  BOOST_CHECK(bn.property(a).type_info == typeid(void));
  BOOST_CHECK_EQUAL(bn.property(a).ptr, nullptr);

  std::set<Arg> seen;
  for (Arg u : bn.vertices()) {
    seen.insert(u);
  }
  BOOST_CHECK_EQUAL(seen.size(), 3);
  BOOST_CHECK(seen.count(a));
  BOOST_CHECK(seen.count(b));
  BOOST_CHECK(seen.count(c));

  // Overwriting a vertex replaces its parent set.
  BOOST_CHECK(!bn.add_vertex(c, {a}));
  BOOST_CHECK_EQUAL(bn.num_edges(), 1);
  BOOST_CHECK_EQUAL(bn.parents(c), Domain({a}));
  BOOST_CHECK_EQUAL(bn.in_degree(c), 1);
  BOOST_CHECK_EQUAL(bn.out_degree(b), 0);
}

BOOST_FIXTURE_TEST_CASE(test_edges_in_out_and_adjacency, Fixture) {
  using edge_type = BayesianNetwork::edge_descriptor;

  BOOST_CHECK(bn.contains(x1, x2));
  BOOST_CHECK(bn.contains(x1, x3));
  BOOST_CHECK(bn.contains(x2, x3));
  BOOST_CHECK(bn.contains(x0, x4));
  BOOST_CHECK(bn.contains(x3, x4));
  BOOST_CHECK(!bn.contains(x4, x3));

  edge_type e = bn.edge(x1, x3);
  BOOST_CHECK(bn.contains(e));
  BOOST_CHECK_EQUAL(e.source(), x1);
  BOOST_CHECK_EQUAL(e.target(), x3);
  BOOST_CHECK(!bn.edge(x4, x3));
  BOOST_CHECK(!bn.edge(x1, x4));

  std::vector<edge_type> expected_in3 = {{x1, x3}, {x2, x3}};
  BOOST_CHECK(range_equal(bn.in_edges(x3), expected_in3));

  std::unordered_set<edge_type> out1 = {{x1, x2}, {x1, x3}};
  for (edge_type oe : bn.out_edges(x1)) {
    BOOST_CHECK_EQUAL(out1.erase(oe), 1);
  }
  BOOST_CHECK(out1.empty());

  std::unordered_set<Arg> children1 = {x2, x3};
  for (Arg child : bn.adjacent_vertices(x1)) {
    BOOST_CHECK_EQUAL(children1.erase(child), 1);
  }
  BOOST_CHECK(children1.empty());
}

BOOST_AUTO_TEST_CASE(test_add_vertex_parent_validation) {
  BayesianNetwork bn;
  Arg a = make_arg("a");
  Arg b = make_arg("b");
  Arg c = make_arg("c");
  Arg z = make_arg("z");

  BOOST_CHECK(bn.add_vertex(a, {}));
  BOOST_CHECK(bn.add_vertex(b, {}));

  BOOST_CHECK_THROW(bn.add_vertex(c, {c}), std::invalid_argument);
  BOOST_CHECK_THROW(bn.add_vertex(c, {a, a}), std::invalid_argument);
  BOOST_CHECK_THROW(bn.add_vertex(c, {z}), std::out_of_range);
  BOOST_CHECK_THROW(bn.add_vertex(b, {a, b}), std::invalid_argument);

  BOOST_CHECK_EQUAL(bn.num_vertices(), 2);
  BOOST_CHECK_EQUAL(bn.num_edges(), 0);
}

BOOST_FIXTURE_TEST_CASE(test_parents_degree_and_counts, Fixture) {
  BOOST_CHECK_EQUAL(bn.parents(x0), Domain());
  BOOST_CHECK_EQUAL(bn.parents(x2), Domain({x1}));
  BOOST_CHECK_EQUAL(bn.parents(x3), Domain({x1, x2}));
  BOOST_CHECK_EQUAL(bn.parents(x4), Domain({x0, x3}));

  BOOST_CHECK_EQUAL(bn.in_degree(x3), 2);
  BOOST_CHECK_EQUAL(bn.out_degree(x1), 2);
  BOOST_CHECK_EQUAL(bn.degree(x3), 3);

  BOOST_CHECK_EQUAL(bn.num_vertices(), 5);
  BOOST_CHECK_EQUAL(bn.num_edges(), 5);
}

BOOST_FIXTURE_TEST_CASE(test_remove_and_clear, Fixture) {
  // x4 is a leaf, so it can be removed directly.
  BOOST_CHECK_EQUAL(bn.remove_vertex(x4), 1);
  BOOST_CHECK(!bn.contains(x4));
  BOOST_CHECK_EQUAL(bn.num_vertices(), 4);
  BOOST_CHECK_EQUAL(bn.num_edges(), 3);

  // Remove incoming edges into x3.
  bn.remove_in_edges(x3);
  BOOST_CHECK_EQUAL(bn.parents(x3), Domain());
  BOOST_CHECK_EQUAL(bn.in_degree(x3), 0);
  BOOST_CHECK_EQUAL(bn.out_degree(x1), 1);
  BOOST_CHECK_EQUAL(bn.out_degree(x2), 0);
  BOOST_CHECK_EQUAL(bn.num_edges(), 1);
  BOOST_CHECK(!bn.contains(x1, x3));
  BOOST_CHECK(!bn.contains(x2, x3));

  bn.clear();
  BOOST_CHECK(bn.empty());
  BOOST_CHECK_EQUAL(bn.num_vertices(), 0);
  BOOST_CHECK_EQUAL(bn.num_edges(), 0);
}

BOOST_FIXTURE_TEST_CASE(test_remove_vertex_preconditions, Fixture) {
  Arg missing = make_arg("missing");
  BOOST_CHECK_THROW(bn.remove_vertex(x1), std::logic_error);
  BOOST_CHECK_EQUAL(bn.remove_vertex(missing), 0);
}

BOOST_FIXTURE_TEST_CASE(test_remove_in_edges_missing_vertex_throws, Fixture) {
  Arg missing = make_arg("missing");
  BOOST_CHECK_THROW(bn.remove_in_edges(missing), std::out_of_range);
}

BOOST_FIXTURE_TEST_CASE(test_markov_network, Fixture) {
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
