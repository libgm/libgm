#ifndef LIBGM_GRAPH_CONNECTED_HPP
#define LIBGM_GRAPH_CONNECTED_HPP

#include <queue>
#include <unordered_set>

namespace libgm {

  /**
   * Returns true if an undirected graph is connected.
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  bool test_connected(const Graph& graph) {
    typedef typename Graph::vertex_type vertex_type;
    if (graph.empty()) { return true; }

    // Search, starting from some arbitrarily chosen vertex
    vertex_type root = *graph.vertices().begin();
    std::unordered_set<vertex_type> visited;
    visited.insert(root);
    std::queue<vertex_type> q;
    q.push(root);

    while (!q.empty() && visited.size() < graph.num_vertices()) {
      vertex_type u = q.front();
      q.pop();
      for (vertex_type v : graph.neighbors(u)) {
        if (!visited.count(v)) {
          visited.insert(v);
          q.push(v);
        }
      }
    }

    return visited.size() == graph.num_vertices();
  }
}

#endif
