#include "elimination_strategies.hpp"

#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/util/bgl.hpp>

#include <ankerl/unordered_dense.h>

#include <algorithm>
#include <iterator>
#include <tuple>

namespace libgm {

ptrdiff_t MinDegreeStrategy::priority(Arg u, const MarkovNetwork& g) const {
  return -static_cast<ptrdiff_t>(g.degree(u));
}

void MinDegreeStrategy::updated(Arg u, const MarkovNetwork& g, std::vector<Arg>& out) const {
  std::copy(g.adjacent_vertices(u).begin(),
            g.adjacent_vertices(u).end(),
            std::back_inserter(out));
}

ptrdiff_t MinFillStrategy::priority(Arg u, const MarkovNetwork& g) const {
  using adjacency_iterator = MarkovNetwork::adjacency_iterator;

  ptrdiff_t new_edges = 0;
  adjacency_iterator it1, end;
  for (std::tie(it1, end) = adjacent_vertices(u, g); it1 != end; ++it1) {
    adjacency_iterator it2 = it1;
    while (++it2 != end) {
      if (!g.contains(*it1, *it2)) {
        ++new_edges;
      }
    }
  }
  return -new_edges;
}

void MinFillStrategy::updated(Arg u, const MarkovNetwork& g, std::vector<Arg>& out) const {
  using adjacency_iterator = MarkovNetwork::adjacency_iterator;

  // It is faster to store the values in a set than to output them
  // multiple times (which causes further priority updates)
  ankerl::unordered_dense::set<Arg> update_set;
  for (Arg v : g.adjacent_vertices(u)) {
    update_set.insert(v);
    adjacency_iterator it, end;
    std::tie(it, end) = adjacent_vertices(v, g);
    update_set.insert(it, end);
  }
  std::copy(update_set.begin(), update_set.end(), std::back_inserter(out));
}

} // namespace libgm
