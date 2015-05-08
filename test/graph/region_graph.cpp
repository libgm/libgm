#define BOOST_TEST_MODULE region_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/region_graph.hpp>

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/universe.hpp>

namespace libgm {
  template class region_graph<basic_domain<std::size_t> >;
}

using namespace libgm;

struct fixture {
  fixture() {
    x = u.new_finite_variables(6, "x", 2);
    root_clusters.push_back({x[1], x[2], x[5]});
    root_clusters.push_back({x[2], x[3], x[5]});
    root_clusters.push_back({x[3], x[4], x[5]});
    root_clusters.push_back({x[4], x[1], x[5]});
    rg.saturated(root_clusters);
  }

  std::size_t find_cluster(domain cluster) {
    cluster.sort();
    std::size_t v = 0;
    for (std::size_t u : rg.vertices()) {
      if (equivalent(rg.cluster(u), cluster)) {
        BOOST_CHECK(v == 0);
        v = u;
      }
    }
    BOOST_CHECK(v != 0);
    return v;
  }

  universe u;
  domain x;
  std::vector<domain> root_clusters;
  region_graph<domain> rg;
};

BOOST_FIXTURE_TEST_CASE(test_validity, fixture) {
  BOOST_CHECK(rg.num_vertices() == 9);
  BOOST_CHECK(rg.num_edges() == 12);

  std::size_t v125 = find_cluster({x[1], x[2], x[5]});
  std::size_t v235 = find_cluster({x[2], x[3], x[5]});
  std::size_t v345 = find_cluster({x[3], x[4], x[5]});
  std::size_t v415 = find_cluster({x[4], x[1], x[5]});
  std::size_t v15 = find_cluster({x[1], x[5]});
  std::size_t v25 = find_cluster({x[2], x[5]});
  std::size_t v35 = find_cluster({x[3], x[5]});
  std::size_t v45 = find_cluster({x[4], x[5]});
  std::size_t v5 = find_cluster({x[5]});

  // check the counting numbers
  for (std::size_t v : rg.vertices()) {
    int number = rg.cluster(v).size() == 2 ? -1 : 1;
    BOOST_CHECK_EQUAL(rg.counting(v), number);
  }

  // check the edges
  BOOST_CHECK(rg.contains(v125, v15));
  BOOST_CHECK(rg.contains(v125, v25));
  BOOST_CHECK(rg.contains(v235, v25));
  BOOST_CHECK(rg.contains(v235, v35));
  BOOST_CHECK(rg.contains(v345, v35));
  BOOST_CHECK(rg.contains(v345, v45));
  BOOST_CHECK(rg.contains(v415, v45));
  BOOST_CHECK(rg.contains(v415, v15));
  BOOST_CHECK(rg.contains(v15, v5));
  BOOST_CHECK(rg.contains(v25, v5));
  BOOST_CHECK(rg.contains(v35, v5));
  BOOST_CHECK(rg.contains(v45, v5));

  // check the separators
  for (directed_edge<std::size_t> e : rg.edges()) {
    BOOST_CHECK_EQUAL(rg.separator(e),
                      rg.cluster(e.source()) & rg.cluster(e.target()));
  }

  // check find_cover
  for (std::size_t v : rg.vertices()) {
    BOOST_CHECK(rg.find_cover(rg.cluster(v)));
  }
}
