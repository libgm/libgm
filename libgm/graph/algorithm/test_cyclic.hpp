#ifndef LIBGM_TEST_CYCLIC_HPP
#define LIBGM_TEST_CYCLIC_HPP

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/graph/vertex_traits.hpp>

namespace libgm {

  /**
   * Test if a directed graph is cyclic.
   *
   * \tparam Graph a directed graph type
   * \return a vertex within a cycle or the null vertex if the graph is acyclic
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  typename Graph::vertex_type test_cyclic(const Graph& graph) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;

    mutable_queue<vertex_type, std::ptrdiff_t> q;
    for (vertex_type v : graph.vertices()) {
      q.push(v, -std::ptrdiff_t(graph.in_degree(v)));
    }

    while (!q.empty()) {
      vertex_type v;
      std::ptrdiff_t indeg;
      std::tie(v, indeg) = q.pop();

      // if all the remaining vertices have parents, then there is a cycle
      if (indeg) {
        return v;
      }

      // remove edges from v to children
      for (vertex_type child : graph.children(v)) {
        q.increment_if_present(child, 1);
      }
    }

    return vertex_traits<vertex_type>::null();
  }

} // namespace libgm

#endif
