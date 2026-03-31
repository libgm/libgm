#define BOOST_TEST_MODULE bayesian_network
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/named_argument.hpp>
#include <libgm/model/bayesian_network.hpp>

#include <set>
#include <stdexcept>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;
using Graph = BayesianNetwork<Arg>;

struct Fixture {
  Fixture() {
    x0 = Arg("x0");
    x1 = Arg("x1");
    x2 = Arg("x2");
    x3 = Arg("x3");
    x4 = Arg("x4");

    v0 = bn.add_vertex(x0, {});
    v1 = bn.add_vertex(x1, {});
    v2 = bn.add_vertex(x2, {x1});
    v3 = bn.add_vertex(x3, {x1, x2});
    v4 = bn.add_vertex(x4, {x0, x3});
  }

  Graph bn;
  Arg x0;
  Arg x1;
  Arg x2;
  Arg x3;
  Arg x4;
  Graph::Vertex* v0 = nullptr;
  Graph::Vertex* v1 = nullptr;
  Graph::Vertex* v2 = nullptr;
  Graph::Vertex* v3 = nullptr;
  Graph::Vertex* v4 = nullptr;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors_and_empty) {
  Graph bn;
  BOOST_CHECK(bn.empty());
  BOOST_CHECK_EQUAL(bn.num_vertices(), 0);
  BOOST_CHECK_EQUAL(bn.num_edges(), 0);

  Arg a("a");
  Arg b("b");

  auto* va = bn.add_vertex(a, {});
  auto* vb = bn.add_vertex(b, {a});
  BOOST_CHECK_EQUAL(bn.vertex(a), va);
  BOOST_CHECK_EQUAL(bn.vertex(b), vb);

  Graph copy(bn);
  BOOST_CHECK(copy.contains(a));
  BOOST_CHECK(copy.contains(copy.vertex(a), copy.vertex(b)));

  Graph assigned;
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
  Graph bn;
  Arg a("a");
  Arg b("b");
  Arg c("c");
  Arg z("z");

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
  Arg z("z");
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

BOOST_FIXTURE_TEST_CASE(test_markov_structure_uses_directed_graph_indices, Fixture) {
  MarkovStructure<Arg> mg = bn.markov_structure();

  BOOST_CHECK_EQUAL(mg.num_vertices(), 5);

  BOOST_CHECK_EQUAL(mg.argument(bn.index(v0)), x0);
  BOOST_CHECK_EQUAL(mg.argument(bn.index(v1)), x1);
  BOOST_CHECK_EQUAL(mg.argument(bn.index(v2)), x2);
  BOOST_CHECK_EQUAL(mg.argument(bn.index(v3)), x3);
  BOOST_CHECK_EQUAL(mg.argument(bn.index(v4)), x4);

  BOOST_CHECK(bn.indices(v0) == std::vector<size_t>({bn.index(v0)}));
  BOOST_CHECK(bn.indices(v2) == std::vector<size_t>({bn.index(v2), bn.index(v1)}));
  BOOST_CHECK(bn.indices(v3) == std::vector<size_t>({bn.index(v3), bn.index(v1), bn.index(v2)}));
  BOOST_CHECK(bn.indices(v4) == std::vector<size_t>({bn.index(v4), bn.index(v0), bn.index(v3)}));

  BOOST_CHECK(bn.arguments(v0) == libgm::Domain<Arg>({x0}));
  BOOST_CHECK(bn.arguments(v2) == libgm::Domain<Arg>({x2, x1}));
  BOOST_CHECK(bn.arguments(v3) == libgm::Domain<Arg>({x3, x1, x2}));
  BOOST_CHECK(bn.arguments(v4) == libgm::Domain<Arg>({x4, x0, x3}));

  BOOST_CHECK(mg.contains(bn.index(v1), bn.index(v2)));
  BOOST_CHECK(mg.contains(bn.index(v1), bn.index(v3)));
  BOOST_CHECK(mg.contains(bn.index(v2), bn.index(v3)));
  BOOST_CHECK(mg.contains(bn.index(v0), bn.index(v3)));
  BOOST_CHECK(mg.contains(bn.index(v0), bn.index(v4)));
  BOOST_CHECK(mg.contains(bn.index(v3), bn.index(v4)));
}
