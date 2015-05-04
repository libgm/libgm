#ifndef LIBGM_ANCESTORS_HPP
#define LIBGM_ANCESTORS_HPP

#include <queue>

namespace libgm {

  /**
   * Computes the ancestors for a set of vertices.
   *
   * \tparam Graph an undirected graph type
   * \tparam Set a set type with elements of type Graph::vertex_type
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Set>
  void ancestors(const Graph& graph, const Set& vertices, Set& result) {
    typedef typename Graph::vertex_type vertex_type;
    std::queue<vertex_type> q;
    for (vertex_type v : vertices) {
      q.push(v);
    }
    while (!q.empty()) {
      vertex_type u = q.front();
      q.pop();
      for (vertex_type v : graph.parents(u)) {
        if (!result.count(v)) {
          result.insert(v);
          q.push(v);
        }
      }
    }
  }

  /**
   * Computes the ancestors for a single vertex.
   *
   * \tparam Graph an undirected graph type
   * \tparam Set a set type with elements of type Graph::vertex_type
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Set>
  void ancestors(const Graph& graph, typename Graph::vertex_type v, Set& result) {
    Set vertices;
    vertices.insert(v);
    ancestors(graph, vertices, result);
  }

} // namespace libgm

#endif
