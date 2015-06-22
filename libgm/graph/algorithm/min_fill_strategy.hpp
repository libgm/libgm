#ifndef LIBGM_MIN_FILL_STRATEGY_HPP
#define LIBGM_MIN_FILL_STRATEGY_HPP

#include <libgm/graph/vertex_traits.hpp>

#include <unordered_set>

namespace libgm {

  /**
   * A class that represents a min-fill elimination strategy.
   * A min-fill elimination strategy gives higher priority that cause
   * smaller number of edges to be introduced as a result of elimination.
   * Whenever a vertex is eliminated, the priority needs to be recomputed
   * for all the vertices within distance 2 of this vertex.
   *
   * This type models the EliminationStrategy concept.
   * \ingroup graph_types
   */
  struct min_fill_strategy {

    //! The priority type associated with each vertex.
    typedef std::ptrdiff_t priority_type;

    //! Computes the priority of a vertex, which is the negative of the fill-in.
    template <typename Graph>
    std::ptrdiff_t priority(typename Graph::vertex_type u, const Graph& g) {
      typedef typename Graph::neighbor_iterator neighbor_iterator;
      std::ptrdiff_t new_edges = 0;
      neighbor_iterator it1, end;
      for (std::tie(it1, end) = g.neighbors(u); it1 != end; ++it1) {
        neighbor_iterator it2 = it1;
        while (++it2 != end) {
          if (!g.contains(*it1, *it2)) {
            ++new_edges;
          }
        }
      }
      return -new_edges;
    }

    //! Stores the vertices whose priority needs to be recomputed to out.
    template <typename Graph, typename OutIt>
    void updated(typename Graph::vertex_type u, const Graph& g, OutIt out) {
      typedef typename Graph::vertex_type vertex_type;
      typedef typename vertex_traits<vertex_type>::hasher hasher;

      // It is faster to store the values in a set than to output them
      // multiple times (which causes further priority updates)
      std::unordered_set<vertex_type, hasher> update_set;
      for (vertex_type v : g.neighbors(u)) {
        update_set.insert(v);
        update_set.insert(g.neighbors(v).begin(), g.neighbors(v).end());
      }
      std::copy(update_set.begin(), update_set.end(), out);
    }

  }; // struct min_fill_strategy

} // namespace libgm

#endif
