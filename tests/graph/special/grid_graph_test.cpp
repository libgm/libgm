#define BOOST_TEST_MODULE grid_graph_util
#include <boost/test/unit_test.hpp>

#include <libgm/graph/special/grid_graph.hpp>

#include <functional>
#include <vector>

using namespace libgm;

BOOST_AUTO_TEST_CASE(creates_expected_grid_connectivity) {
  std::vector<std::vector<Arg>> args(3, std::vector<Arg>(4));

  MarkovNetworkT<int, double> graph = make_grid_graph<int, double>(3, 4, make_argument);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      args[row][col] = make_argument(row, col);
    }
  }

  BOOST_CHECK_EQUAL(graph.num_vertices(), 12);
  BOOST_CHECK_EQUAL(graph.num_edges(), 17);

  BOOST_CHECK_EQUAL(graph.degree(args[0][0]), 2);
  BOOST_CHECK_EQUAL(graph.degree(args[1][1]), 4);

  for (size_t row1 = 0; row1 < 3; ++row1) {
    for (size_t col1 = 0; col1 < 4; ++col1) {
      for (size_t row2 = 0; row2 < 3; ++row2) {
        for (size_t col2 = 0; col2 < 4; ++col2) {
          const size_t dr = row1 > row2 ? row1 - row2 : row2 - row1;
          const size_t dc = col1 > col2 ? col1 - col2 : col2 - col1;
          const bool should_be_connected = (dr + dc) == 1;
          BOOST_CHECK_EQUAL(
              graph.contains(args[row1][col1], args[row2][col2]),
              should_be_connected);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(empty_when_dimension_is_zero) {
  int call_count = 0;
  std::function<Arg(size_t, size_t)> counted_make_argument = [&](size_t row, size_t col) {
    ++call_count;
    return make_argument(row, col);
  };

  MarkovNetworkT<int, int> graph1 = make_grid_graph<int, int>(0, 5, counted_make_argument);
  MarkovNetworkT<int, int> graph2 = make_grid_graph<int, int>(4, 0, counted_make_argument);

  BOOST_CHECK(graph1.empty());
  BOOST_CHECK(graph2.empty());
  BOOST_CHECK_EQUAL(call_count, 0);
}
