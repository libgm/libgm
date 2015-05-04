#define BOOST_TEST_MODULE graph_traversal
#include <boost/test/unit_test.hpp>

#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/directed_graph.hpp>
//#include <libgm/graph/directed_multigraph.hpp>

#include <boost/mpl/list.hpp>

#include <utility>
#include <vector>

#include "predicates.hpp"

using namespace libgm;

typedef boost::mpl::list<
  directed_graph<int, int, double>//,
  //  directed_multigraph<int, int, double>
> graph_types;

typedef std::pair<int, int> vpair;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_simple, Graph, graph_types) {
  std::vector<vpair> vpairs = 
    {vpair(5, 2), vpair(1, 2), vpair(1, 3), vpair(1, 7), 
     vpair(2, 3), vpair(3, 4)};
  Graph g(vpairs);
  std::vector<int> order;
  partial_order_traversal(g, [&](int v) { order.push_back(v); });
  BOOST_CHECK(is_partial_vertex_order(order, g));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_multi_edges, Graph, graph_types) {
  std::vector<vpair> vpairs =
    {vpair(5, 2), vpair(1, 2), vpair(1, 3), vpair(1, 7), 
     vpair(2, 3), vpair(3, 4), vpair(1, 2)};
  Graph g(vpairs);
  std::vector<int> order;
  partial_order_traversal(g, [&](int v) { order.push_back(v); });
  BOOST_CHECK(is_partial_vertex_order(order, g));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_cycle, Graph, graph_types) {
  std::vector<vpair> vpairs =
    {vpair(5, 2), vpair(1, 2), vpair(1, 3), vpair(1, 7), 
     vpair(2, 3), vpair(3, 4), vpair(4, 1)};
  Graph g(vpairs);
  BOOST_CHECK_THROW(partial_order_traversal(g, [](int v) { }),
                    std::invalid_argument);
}
