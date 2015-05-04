#ifndef LIBGM_TREE_TRAVERSAL_HPP
#define LIBGM_TREE_TRAVERSAL_HPP

#include <libgm/functional/output_iterator_assign.hpp>

#include <functional>
#include <iterator>
#include <stdexcept>
#include <queue>
#include <vector>

namespace libgm {

  /**
   * Performs a pre-order traversal of a tree starting at the given root,
   * invoking a visitor at the same time as each edge is traversed.
   * The order in which the visitor is invoked is consistent
   * with a breadth-first or depth-first traversal from v.
   *
   * \param graph
   *        The graph being traversed. This graph must be singly connected
   *        and be either undirected or bidirectional.
   * \param root
   *        A vertex of the graph. The traversal is started at this vertex.
   * \param visitor
   *        The visitor that is invoked to each edge traversed.
   *        If the graph is biconnected, then this visitor is applied
   *        only to edges directed away from the root vertex.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  void pre_order_traversal(
      const Graph& graph, 
      typename Graph::vertex_type root,
      std::function<void(const typename Graph::edge_type&)> visitor) {
    typedef typename Graph::edge_type edge_type;
    
    // create a queue initialized all edges outgoing from the root
    std::queue<edge_type> queue;
    for (edge_type e : graph.out_edges(root)) {
      queue.push(e);
    }

    // process edges recursively until there are no more or we detected cycle
    std::size_t nvisited = 0;
    while (!queue.empty()) {
      // visit the incoming edge
      edge_type in = queue.front();
      queue.pop();
      visitor(in);
      if (++nvisited >= graph.num_vertices()) {
        throw std::invalid_argument("tree traversal: detected a cycle");
      }
      // enqueue the outgoing edges of the target, except those leading
      // back to the source
      for (edge_type out : graph.out_edges(in.target())) {
        if (out.target() != in.source()) {
          queue.push(out);
        }
      }
    }
  }

  /**
   * Performs a post-order traversal of a tree, starting at the given root.
   * The given edge visitor is applied to each edge during the traversal.
   * The reverse of the order in which the visitor is applied to the edges
   * is consistent with a breadth-first or depth-first traversal from root.
   *
   * \param graph
   *        The graph on whose edges the visitor is applied. This
   *        graph must be singly connected, and it must be either
   *        undirected or bidirectional.
   * \param root
   *        A vertex of the graph. The traversal is started at this vertex.
   * \param visitor
   *        The visitor that is applied to each edge of the graph.
   *        If the graph is biconnected, then this visitor is applied
   *        only to edges directed toward from the start vertex.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  void post_order_traversal(
      const Graph& graph,
      typename Graph::vertex_type root,
      std::function<void(const typename Graph::edge_type&)> visitor) {
    typedef typename Graph::edge_type edge_type;

    // first compute a pre-order traversal of the edges starting at root
    std::vector<edge_type> edges;
    pre_order_traversal(graph, root,
                        make_output_iterator_assign(std::back_inserter(edges)));

    // now visit the reverse of these edges in the reverse order
    for (size_t i = edges.size(); i > 0; --i) {
      visitor(graph.reverse(edges[i-1]));
    }
  }

  /**
   * Visits each (directed) edge of a tree graph once in a traversal
   * such that each \f$v \rightarrow w\f$ is visited after all edges
   * \f$u \rightarrow v\f$ (with \f$u \neq w\f$) are visited.  Orders
   * of this type are said to satisfy the "message passing protocol"
   * (MPP).
   *
   * \param graph
   *        The graph on whose edges the visitor is applied. This
   *        graph must be singly connected, and it must be either
   *        undirected or bidirectional.
   * \param root
   *        A vertex of the graph. The traversal is started at this vertex.
   *        Can be a null vertex to denote arbitrary root.
   * \param visitor
   *        The visitor that is applied to each edge of the graph.
   *        If the graph is biconnected, then this visitor is applied
   *        only to edges directed toward from the start vertex.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  void mpp_traversal(
      const Graph& graph,
      typename Graph::vertex_type root,
      std::function<void(const typename Graph::edge_type&)> visitor) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;

    // if the root was not specified, choose one arbitrarily
    if (root == vertex_type()) {
      if (graph.empty()) return;
      root = *graph.vertices().begin();
    }

    // collect towards the root and then distribute from the root
    post_order_traversal(graph, root, visitor);
    pre_order_traversal(graph, root, visitor);
  }

} // namespace libgm

#endif
