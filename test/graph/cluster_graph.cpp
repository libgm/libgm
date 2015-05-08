#define BOOST_TEST_MODULE cluster_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/cluster_graph.hpp>

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/universe.hpp>

#include <functional>

#include "../predicates.hpp"

namespace libgm {
  template class cluster_graph<domain>;
}

using namespace libgm;

struct fixture {
  fixture()
    : v(u.new_finite_variables(6, "v", 2)) {
    cg.add_cluster(1, {v[0], v[1]});
    cg.add_cluster(2, {v[1], v[2], v[3]});
    cg.add_cluster(3, {v[2], v[3], v[4]});
    cg.add_cluster(4, {v[3], v[5]});
    cg.add_edge(1, 2);
    cg.add_edge(2, 3);
    cg.add_edge(2, 4);
  }

  universe u;
  domain v;
  cluster_graph<domain> cg;
};

BOOST_FIXTURE_TEST_CASE(test_properties, fixture) {
  BOOST_CHECK(cg.connected());
  BOOST_CHECK(cg.tree());
  BOOST_CHECK(cg.running_intersection());

  cg.remove_edge(2, 4);
  BOOST_CHECK(!cg.connected());
  BOOST_CHECK(!cg.running_intersection());
}

BOOST_FIXTURE_TEST_CASE(test_copy, fixture) {
  cluster_graph<domain> cg2(cg);
  BOOST_CHECK(cg2.connected());
  BOOST_CHECK(cg2.tree());
  BOOST_CHECK(cg2.running_intersection());
  BOOST_CHECK_EQUAL(cg, cg2);
}

/*
BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(cg, u));
}
*/

BOOST_AUTO_TEST_CASE(test_triangulated) {
  // Build the graph. This graph must have no self-loops or parallel edges.
  typedef std::pair<std::size_t, std::size_t> vpair;
  std::vector<vpair> vpairs =
    {vpair(6, 2), vpair(1, 2), vpair(1, 3), vpair(1, 5),
     vpair(2, 3), vpair(3, 4), vpair(4, 6), vpair(4, 1)};
  undirected_graph<std::size_t> g(vpairs);
  typedef basic_domain<std::size_t> domain_type;

  // Build a junction tree using the min-degree strategy
  cluster_graph<domain_type> jt;
  jt.triangulated(g, min_degree_strategy());
  BOOST_CHECK(jt.connected());
  BOOST_CHECK(jt.tree());
  BOOST_CHECK(jt.running_intersection());
  std::cout << jt << std::endl;

  // Verify the cliques and separators
  BOOST_CHECK_EQUAL(jt.num_vertices(), 3);
  BOOST_CHECK_EQUAL(jt.num_edges(), 2);
  BOOST_CHECK(equivalent(jt.cluster(1), domain_type{1, 5}));
  BOOST_CHECK(equivalent(jt.cluster(2), domain_type{6, 2, 4}));
  BOOST_CHECK(equivalent(jt.cluster(3), domain_type{1, 2, 3, 4}));
  BOOST_CHECK(jt.contains(1, 3));
  BOOST_CHECK(jt.contains(2, 3));
  BOOST_CHECK(equivalent(jt.separator(1, 3), domain_type{1}));
  BOOST_CHECK(equivalent(jt.separator(2, 3), domain_type{2, 4}));

  // Check the subtree cover for the set {5, 6}
  jt.mark_subtree_cover({6, 5}, true);
  BOOST_CHECK(jt.marked(1));
  BOOST_CHECK(jt.marked(2));
  BOOST_CHECK(jt.marked(3));
  BOOST_CHECK(jt.marked(jt.edge(1, 3)));
  BOOST_CHECK(jt.marked(jt.edge(2, 3)));

  // Check the subtree cover for the set {1}
  jt.mark_subtree_cover({1}, true);
  BOOST_CHECK(jt.marked(1) != jt.marked(3));
  BOOST_CHECK(!jt.marked(2));
  BOOST_CHECK(!jt.marked(jt.edge(1, 3)));
  BOOST_CHECK(!jt.marked(jt.edge(2, 3)));

  // Make some changes and check if still valid
  std::size_t v6 = jt.find_cluster_cover({6});
  std::size_t v67 = jt.add_cluster({6, 7});
  jt.add_edge(v6, v67);
  BOOST_CHECK_EQUAL(jt.num_vertices(), 4);
  BOOST_CHECK_EQUAL(jt.num_edges(), 3);
  BOOST_CHECK(jt.connected());
  BOOST_CHECK(jt.tree());
  BOOST_CHECK(jt.running_intersection());
}
