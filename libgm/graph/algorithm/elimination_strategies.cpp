#include "elimination_strategies.hpp"

#include <libgm/graph/vector_graph.hpp>

namespace libgm {

ptrdiff_t MinDegreeStrategy::priority(size_t u, const VectorGraph& g) const {
  return -static_cast<ptrdiff_t>(g.degree(u));
}

void MinDegreeStrategy::updated(size_t u,
                                const VectorGraph& g,
                                std::vector<size_t>& out) const {
  for (size_t v : g.adjacent_vertices(u)) {
    out.push_back(v);
  }
}

ptrdiff_t MinFillStrategy::priority(size_t u, const VectorGraph& g) const {
  ptrdiff_t new_edges = 0;
  auto neighbors = g.adjacent_vertices(u);
  for (auto it1 = neighbors.begin(); it1 != neighbors.end(); ++it1) {
    auto it2 = it1;
    while (++it2 != neighbors.end()) {
      if (!g.contains(*it1, *it2)) {
        ++new_edges;
      }
    }
  }
  return -new_edges;
}

void MinFillStrategy::updated(size_t u,
                              const VectorGraph& g,
                              std::vector<size_t>& out) const {
  for (size_t v : g.adjacent_vertices(u)) {
    out.push_back(v);
    for (size_t w : g.adjacent_vertices(v)) {
      out.push_back(w);
    }
  }
}

} // namespace libgm
