#ifndef LIBGM_TRIANGULATE_HPP
#define LIBGM_TRIANGULATE_HPP

#include <libgm/datastructure/set_index.hpp>
#include <libgm/graph/algorithm/eliminate.hpp>
#include <libgm/graph/algorithm/min_degree_strategy.hpp>

#include <functional>

namespace libgm {

  /**
   * Computes a triangulation of a graph using greedy vertex elimination.
   * A triangulation is represented as a collection of cliques whose union
   * includes all the edges (and vertices) of the graph.
   *
   * \tparam Container
   *         The type that can represent the clique. This type must be
   *         specified explicitly.
   * \tparam Graph
   *         The graph type (typically an undirected graph)
   * \tparam Strategy
   *         A type that model the EliminationStrategy concept.
   * \param g
   *        The graph to be triangulated; it must have no self-loops
   *        or parallel edges, and its edges must be undirected or
   *        bidirected. This procedure removes all edges from the graph.
   * \param visitor
   *        A function object that gets invoked for each clique.
   *        This object can accept both lvalue and rvalue references
   *        to Container.
   * \param elim_strategy
   *        The elimination strategy used to determine the elimination order.
   *
   * \ingroup graph_algorithms
   */
  template <typename Container,
            typename Graph,
            typename Strategy = min_degree_strategy>
  void triangulate(Graph& g,
                   std::function<void(Container&&)> visitor,
                   Strategy elim_strategy = Strategy()) {
    eliminate(g, [&](typename Graph::vertex_type v) {
        Container clique({v});
        clique.insert(clique.end(), g.neighbors(v).begin(), g.neighbors(v).end());
        visitor(std::move(clique));
      }, elim_strategy);
  }

  /**
   * Computes a triangulation of a graph using greedy vertex elimination.
   * A triangulation is represented as a collection of cliques whose union
   * includes all the edges (and vertices) of the graph. Unlike triangulate,
   * this function only outputs maximal cliques.
   *
   * \tparam Container
   *         The type that can represent the clique. This type must be
   *         specified explicitly.
   * \tparam Graph
   *         The graph type (typically an undirected graph)
   * \tparam Strategy
   *         A type that model the EliminationStrategy concept.
   * \param g
   *        The graph to be triangulated; it must have no self-loops
   *        or parallel edges, and its edges must be undirected or
   *        bidirected. This procedure removes all edges from the graph.
   * \param visitor
   *        A function object that gets invoked for each clique.
   *        This object can accept both lvalue and rvalue references
   *        to Container.
   * \param elim_strategy
   *        The elimination strategy used to determine the elimination order.
   *
   * \ingroup graph_algorithms
   */
  template <typename Container,
            typename Graph,
            typename Strategy = min_degree_strategy>
  void triangulate_maximal(Graph& g,
                           std::function<void(Container&&)> visitor,
                           Strategy elim_strategy = Strategy()) {
    set_index<size_t, Container> clique_index;
    std::size_t id = 0;
    eliminate(g, [&](typename Graph::vertex_type v) {
        Container clique({v});
        clique.insert(clique.end(), g.neighbors(v).begin(), g.neighbors(v).end());
        if (clique_index.is_maximal(clique)) {
          clique_index.insert(id++, clique);
          visitor(std::move(clique));
        }
      }, elim_strategy);
  }

} // namespace libgm

#endif
