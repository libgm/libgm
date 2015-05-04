#ifndef LIBGM_GRAPH_TRAVERSAL_HPP
#define LIBGM_GRAPH_TRAVERSAL_HPP

#include <libgm/global.hpp>

#include <queue>
#include <unordered_map>

namespace libgm {

  /**
   * Visits each vertex of a directed acyclic graph once in a traversal
   * such that each \f$v\f$ is visited after all nodes \f$u\f$ with
   * \f$u \rightarrow v\f$ are visited.
   *
   * \tparam Graph a directed graph type
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  void partial_order_traversal(
      const Graph& graph,
      std::function<void(typename Graph::vertex_type)> visitor) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;

    // Split the vertices to those without and with parents
    std::queue<vertex_type> q;
    std::unordered_map<vertex_type, size_t> in_degree;
    in_degree.reserve(graph.num_vertices());
    for (vertex_type v : graph.vertices()) {
      if (graph.in_degree(v) == 0) {
        q.push(v);
      } else {
        in_degree[v] = graph.in_degree(v);
      }
    }

    // Process the remaining vertices in partial order
    while (!q.empty()) {
      vertex_type u = q.front();
      visitor(u);
      q.pop();
      for (vertex_type v : graph.children(u)) {
        if (--in_degree[v] == 0) {
          in_degree.erase(v);
          q.push(v);
        }
      }
    }

    // Check if there were any loops
    if (!in_degree.empty()) {
      throw std::invalid_argument("The graph contains directed loops");
    }
  }

} // namespace libgm

#endif
