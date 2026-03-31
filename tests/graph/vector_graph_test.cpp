#define BOOST_TEST_MODULE vector_graph
#include <boost/test/unit_test.hpp>

#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/vector_graph.hpp>

#include <stdexcept>
#include <vector>

using namespace libgm;

namespace {

struct SelectVertexStrategy : EliminationStrategy {
  explicit SelectVertexStrategy(size_t selected)
    : selected(selected) {}

  ptrdiff_t priority(size_t u, const VectorGraph&) const override {
    return u == selected ? 1 : 0;
  }

  void updated(size_t u, const VectorGraph& g, std::vector<size_t>& out) const override {
    for (size_t v : g.adjacent_vertices(u)) {
      out.push_back(v);
    }
  }

  size_t selected;
};

} // namespace

BOOST_AUTO_TEST_CASE(test_empty_and_vertices) {
  VectorGraph g;
  BOOST_CHECK(g.empty());
  BOOST_CHECK_EQUAL(g.num_vertices(), 0);

  size_t v0 = g.add_vertex();
  size_t v1 = g.add_vertex();
  BOOST_CHECK_EQUAL(v0, 0);
  BOOST_CHECK_EQUAL(v1, 1);
  BOOST_CHECK_EQUAL(g.num_vertices(), 2);
  BOOST_CHECK(!g.empty());

  std::vector<size_t> expected = {0, 1};
  std::vector<size_t> actual(g.vertices().begin(), g.vertices().end());
  BOOST_CHECK(actual == expected);
}

BOOST_AUTO_TEST_CASE(test_add_clique_builds_sorted_adjacency) {
  VectorGraph g(5);
  g.add_clique({3, 1, 4});

  BOOST_CHECK(g.contains(1, 3));
  BOOST_CHECK(g.contains(1, 4));
  BOOST_CHECK(g.contains(3, 4));
  BOOST_CHECK(!g.contains(0, 1));

  std::vector<size_t> neighbors1(g.adjacent_vertices(1).begin(), g.adjacent_vertices(1).end());
  std::vector<size_t> neighbors3(g.adjacent_vertices(3).begin(), g.adjacent_vertices(3).end());
  std::vector<size_t> neighbors4(g.adjacent_vertices(4).begin(), g.adjacent_vertices(4).end());

  BOOST_CHECK(neighbors1 == std::vector<size_t>({3, 4}));
  BOOST_CHECK(neighbors3 == std::vector<size_t>({1, 4}));
  BOOST_CHECK(neighbors4 == std::vector<size_t>({1, 3}));
}

BOOST_AUTO_TEST_CASE(test_add_clique_performs_union_and_deduplicates) {
  VectorGraph g(4);
  g.add_clique({2, 0, 1});
  g.add_clique({1, 2, 3, 3});

  BOOST_CHECK_EQUAL(g.num_vertices(), 4);
  BOOST_CHECK(g.contains(0, 1));
  BOOST_CHECK(g.contains(0, 2));
  BOOST_CHECK(g.contains(1, 2));
  BOOST_CHECK(g.contains(1, 3));
  BOOST_CHECK(g.contains(2, 3));
  BOOST_CHECK(!g.contains(0, 3));

  std::vector<size_t> neighbors1(g.adjacent_vertices(1).begin(), g.adjacent_vertices(1).end());
  std::vector<size_t> neighbors2(g.adjacent_vertices(2).begin(), g.adjacent_vertices(2).end());
  std::vector<size_t> neighbors3(g.adjacent_vertices(3).begin(), g.adjacent_vertices(3).end());

  BOOST_CHECK(neighbors1 == std::vector<size_t>({0, 2, 3}));
  BOOST_CHECK(neighbors2 == std::vector<size_t>({0, 1, 3}));
  BOOST_CHECK(neighbors3 == std::vector<size_t>({1, 2}));
}

BOOST_AUTO_TEST_CASE(test_edge_and_degree_views) {
  VectorGraph g(3);
  g.add_clique({0, 2, 1});

  auto e01 = g.edge(0, 1);
  auto e10 = g.edge(1, 0);
  BOOST_CHECK(e01);
  BOOST_CHECK(e10);
  BOOST_CHECK_EQUAL(e01.source(), 0);
  BOOST_CHECK_EQUAL(e01.target(), 1);
  BOOST_CHECK(!g.edge(0, 3));

  std::vector<VectorGraph::edge_descriptor> out1(g.out_edges(1).begin(), g.out_edges(1).end());
  std::vector<VectorGraph::edge_descriptor> in1(g.in_edges(1).begin(), g.in_edges(1).end());
  std::vector<VectorGraph::edge_descriptor> expected_out = {{1, 0}, {1, 2}};
  std::vector<VectorGraph::edge_descriptor> expected_in = {{0, 1}, {2, 1}};

  BOOST_CHECK(out1 == expected_out);
  BOOST_CHECK(in1 == expected_in);
  BOOST_CHECK_EQUAL(g.degree(1), 2);
}

BOOST_AUTO_TEST_CASE(test_remove_edge_marks_erased_and_iterators_skip_it) {
  VectorGraph g(3);
  g.add_clique({0, 1, 2});

  BOOST_CHECK_EQUAL(g.remove_edge(0, 1), 1);
  BOOST_CHECK_EQUAL(g.remove_edge(0, 1), 0);
  BOOST_CHECK(!g.contains(0, 1));
  BOOST_CHECK(g.contains(0, 2));
  BOOST_CHECK(g.contains(1, 2));

  std::vector<size_t> neighbors0(g.adjacent_vertices(0).begin(), g.adjacent_vertices(0).end());
  std::vector<size_t> neighbors1(g.adjacent_vertices(1).begin(), g.adjacent_vertices(1).end());
  BOOST_CHECK(neighbors0 == std::vector<size_t>({2}));
  BOOST_CHECK(neighbors1 == std::vector<size_t>({2}));
}

BOOST_AUTO_TEST_CASE(test_add_clique_restores_erased_edges) {
  VectorGraph g(3);
  g.add_clique({0, 1, 2});
  BOOST_CHECK_EQUAL(g.remove_edge(0, 1), 1);

  g.add_clique({0, 1});
  BOOST_CHECK(g.contains(0, 1));
  BOOST_CHECK(g.contains(0, 2));
  BOOST_CHECK(g.contains(1, 2));

  std::vector<size_t> neighbors0(g.adjacent_vertices(0).begin(), g.adjacent_vertices(0).end());
  BOOST_CHECK(neighbors0 == std::vector<size_t>({1, 2}));
}

BOOST_AUTO_TEST_CASE(test_clear_vertex_removes_incident_edges) {
  VectorGraph g(4);
  g.add_clique({0, 1, 2});
  g.add_clique({1, 3});

  BOOST_CHECK_EQUAL(g.clear_vertex(1), 3);
  BOOST_CHECK(!g.contains(0, 1));
  BOOST_CHECK(!g.contains(1, 2));
  BOOST_CHECK(!g.contains(1, 3));
  BOOST_CHECK(g.contains(0, 2));

  std::vector<size_t> neighbors1(g.adjacent_vertices(1).begin(), g.adjacent_vertices(1).end());
  std::vector<size_t> neighbors0(g.adjacent_vertices(0).begin(), g.adjacent_vertices(0).end());
  std::vector<size_t> neighbors2(g.adjacent_vertices(2).begin(), g.adjacent_vertices(2).end());
  std::vector<size_t> neighbors3(g.adjacent_vertices(3).begin(), g.adjacent_vertices(3).end());

  BOOST_CHECK(neighbors1.empty());
  BOOST_CHECK(neighbors0 == std::vector<size_t>({2}));
  BOOST_CHECK(neighbors2 == std::vector<size_t>({0}));
  BOOST_CHECK(neighbors3.empty());
}

BOOST_AUTO_TEST_CASE(test_add_clique_requires_existing_vertices) {
  VectorGraph g;
  g.add_vertex();
  g.add_vertex();
  g.add_vertex();

  BOOST_CHECK_THROW(g.add_clique({2, 5}), std::out_of_range);
  BOOST_CHECK_EQUAL(g.num_vertices(), 3);
  BOOST_CHECK(!g.contains(2, 5));
}

BOOST_AUTO_TEST_CASE(test_eliminate_adds_fill_in_and_clears_vertex) {
  VectorGraph g(4);
  g.add_clique({0, 1});
  g.add_clique({1, 2});
  g.add_clique({1, 3});

  std::vector<size_t> order;
  std::vector<size_t> second_neighbors;
  g.eliminate(SelectVertexStrategy(1), [&](size_t u) {
    if (order.size() == 1) {
      second_neighbors.assign(g.adjacent_vertices(u).begin(), g.adjacent_vertices(u).end());
    }
    order.push_back(u);
  });

  BOOST_CHECK_EQUAL(order.front(), 1);
  BOOST_CHECK_EQUAL(second_neighbors.size(), 2);

  std::vector<size_t> neighbors1(g.adjacent_vertices(1).begin(), g.adjacent_vertices(1).end());
  BOOST_CHECK(neighbors1.empty());
}

BOOST_AUTO_TEST_CASE(test_min_fill_eliminates_star_center_last) {
  VectorGraph g(4);
  g.add_clique({0, 1});
  g.add_clique({0, 2});
  g.add_clique({0, 3});

  std::vector<size_t> order;
  MinFillStrategy strategy;
  g.eliminate(strategy, [&](size_t u) {
    order.push_back(u);
  });

  BOOST_CHECK_EQUAL(order.size(), 4);
  BOOST_CHECK_EQUAL(order.back(), 0);
}
