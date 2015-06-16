#ifndef LIBGM_ELIMINATE_HPP
#define LIBGM_ELIMINATE_HPP

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/graph/algorithm/make_clique.hpp>
#include <libgm/graph/algorithm/min_degree_strategy.hpp>

#include <iterator>
#include <vector>

namespace libgm {

  /**
   * Runs the vertex elimination algorithm on a graph. The algorithm eliminates
   * each node from the graph; eliminating a node involves connecting the node's
   * neighbors into a new clique and then removing the node from the graph.
   * The nodes are eliminated greedily in the order specified by the elimination
   * strategy.
   *
   * \tparam Graph
   *         The graph type (typically an undirected_graph).
   * \tparam Visitor
   *         A type that modes the VertexVisitor concept.
   * \tparam Strategy
   *         A type that models the EliminationStrategy concept.
   * \param graph
   *        The graph whose nodes are eliminated; it must have no
   *        self-loops or parallel edges, and its edges must be
   *        undirected or bidirected.
   * \param vertex_visitor
   *        The visitor which is applied to each vertex before the vertex is
   *        eliminated.
   * \param elim_strategy
   *        the elimination strategy used to determine the elimination order
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph,
            typename Visitor,
            typename Strategy = min_degree_strategy>
  void eliminate(Graph& graph,
                 Visitor vertex_visitor,
                 Strategy elim_strategy = Strategy()) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Strategy::priority_type priority_type;

    // Make a priority queue of vertex indices, ordered by priority
    mutable_queue<vertex_type, priority_type> pq;
    for (vertex_type v : graph.vertices()) {
      pq.push(v, elim_strategy.priority(v, graph));
    }

    // Start the vertex elimination loop
    std::vector<vertex_type> recompute_priority;
    while (!pq.empty()) {
      // get the next vertex to be eliminated
      vertex_type u = pq.pop().first;
      // find out vertices whose priority may change
      recompute_priority.clear();
      elim_strategy.updated(u, graph, std::back_inserter(recompute_priority));
      // eliminate the vertex
      vertex_visitor(u);
      make_clique(graph, graph.neighbors(u));
      graph.remove_edges(u);
      // update the priorities
      for (vertex_type v : recompute_priority) {
        if (v != u) {
          pq.update(v, elim_strategy.priority(v, graph));
        }
      }
    }
  }

} // namespace libgm

#endif
