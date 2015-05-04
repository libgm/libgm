#ifndef LIBGM_SUBGRAPH_HPP
#define LIBGM_SUBGRAPH_HPP

#include <libgm/global.hpp>

namespace libgm {
  
  /**
   * Computes a subgraph of a graph over the given range of vertices.
   *
   * \tparam Graph a graph type (directed or undirected)
   * \tparam VertexRange a collection of vertices
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename VertexRange>
  void subgraph(const Graph& graph,
                const VertexRange& new_vertices,
                Graph& new_graph) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;
    
    new_graph.clear();
    for (vertex_type v : new_vertices) {
      new_graph.add_vertex(v, graph[v]);
    }

    for (vertex_type v : new_vertices) {
      for (edge_type e : graph.out_edges(v)) {
        if (new_graph.contains(e.target())) {
          new_graph.add_edge(e.source(), e.target(), graph[e]);
        }
      }
    }
  }

} // namespace libgm

#endif
