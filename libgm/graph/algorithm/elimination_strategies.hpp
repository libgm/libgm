#pragma once

#include <libgm/graph/elimination_strategy.hpp>

namespace libgm {

/**
 * A class that represents a min-degree elimination strategy.
 * The min-degree strategy gives higher priority to the vertices
 * with fewer neighbors. Whenever a vertex is eliminated, the
 * priorities of its neighbors need to be recomputed.
 *
 */
struct MinDegreeStrategy : EliminationStrategy {
  ptrdiff_t priority(size_t u, const VectorGraph& g) const override;

  void updated(size_t u, const VectorGraph& g, std::vector<size_t>& out) const override;
};

/**
 * A class that represents a min-fill elimination strategy.
 * A min-fill elimination strategy gives higher priority that cause
 * smaller number of edges to be introduced as a result of elimination.
 * Whenever a vertex is eliminated, the priority needs to be recomputed
 * for all the vertices within distance 2 of this vertex.
 */
struct MinFillStrategy : EliminationStrategy {
  ptrdiff_t priority(size_t u, const VectorGraph& g) const override;

  void updated(size_t u, const VectorGraph& g, std::vector<size_t>& out) const override;
};

} // namespace libgm
