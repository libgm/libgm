#pragma once

#include <libgm/graph/markov_network.hpp>

#include <algorithm>

namespace libgm {

/**
 * A class that represents a min-degree elimination strategy.
 * The min-degree strategy gives higher priority to the vertices
 * with fewer neighbors. Whenever a vertex is eliminated, the
 * priorities of its neighbors need to be recomputed.
 *
 * \ingroup graph_types
 */
struct MinDegreeStrategy : MarkovNetwork::EliminationStrategy {
  /// Computes the priority of a vertex, which is its negative degree.
  ptrdiff_t priority(Arg u, const MarkovNetwork& g) const override;

  /// Stores the vertices whose priority needs to be recomputed.
  void updated(Arg u, const MarkovNetwork& g, std::vector<Arg>& out) const override;
}; // struct MinDegreeStrategy

/**
 * A class that represents a min-fill elimination strategy.
 * A min-fill elimination strategy gives higher priority that cause
 * smaller number of edges to be introduced as a result of elimination.
 * Whenever a vertex is eliminated, the priority needs to be recomputed
 * for all the vertices within distance 2 of this vertex.
 *
 */
struct MinFillStrategy : MarkovNetwork::EliminationStrategy {
  using adjacency_iterator = MarkovNetwork::adjacency_iterator;

  /// Computes the priority of a vertex, which is the negative of the fill-in.
  ptrdiff_t priority(Arg u, const MarkovNetwork& g) const override;

  /// Stores the vertices whose priority needs to be recomputed.
  void updated(Arg u, const MarkovNetwork& g, std::vector<Arg>& out) const override;
};

} // namespace libgm

#endif
