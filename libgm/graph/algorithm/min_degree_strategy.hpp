#ifndef LIBGM_MIN_DEGREE_STRATEGY_HPP
#define LIBGM_MIN_DEGREE_STRATEGY_HPP

#include <algorithm>

namespace libgm {

  /**
   * A class that represents a min-degree elimination strategy.
   * The min-degree strategy gives higher priority to the vertices
   * with fewer neighbors. Whenever a vertex is eliminated, the
   * priorities of its neighbors need to be recomputed.
   *
   * This type models the EliminationStrategy concept.
   * \ingroup graph_types
   */
  struct min_degree_strategy {

    //! The priority type associated with each vertex.
    typedef ptrdiff_t priority_type;

    //! Computes the priority of a vertex, which is its negative degree.
    template <typename Graph>
    ptrdiff_t priority(typename Graph::vertex_type u, const Graph& g) {
      return -static_cast<ptrdiff_t>(g.out_degree(u));
    }

    //! Stores the vertices whose priority needs to be recomputed to out.
    template <typename Graph, typename OutIt>
    void updated(typename Graph::vertex_type u, const Graph& g, OutIt out) {
      std::copy(g.neighbors(u).begin(), g.neighbors(u).end(), out);
    }

  }; // struct min_degree_strategy

} // namespace libgm

#endif
