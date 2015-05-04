#ifndef LIBGM_MAKE_CLIQUE_HPP
#define LIBGM_MAKE_CLIQUE_HPP

namespace libgm {
  
  /**
   * Adds edges among all vertices in an undirected graph.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Range>
  void make_clique(Graph& graph, const Range& vertices) {
    for (auto it1 = vertices.begin(), end = vertices.end(); it1 != end; ++it1) {
      graph.add_vertex(*it1);
      auto it2 = it1;
      while (++it2 != end) {
        graph.add_edge(*it1, *it2);
      }
    }
  }

} // namespace libgm

#endif
