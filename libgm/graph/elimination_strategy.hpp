#pragma once

#include <cstddef>
#include <vector>

namespace libgm {

class VectorGraph;

struct EliminationStrategy {
  virtual ~EliminationStrategy() = default;

  virtual ptrdiff_t priority(size_t u, const VectorGraph& g) const = 0;

  /**
   * Stores vertices whose priority may need to be recomputed after eliminating
   * `u`. Implementations may output duplicates; `VectorGraph::eliminate()`
   * deduplicates the updates internally.
   */
  virtual void updated(size_t u,
                       const VectorGraph& g,
                       std::vector<size_t>& output) const = 0;
};

} // namespace libgm
