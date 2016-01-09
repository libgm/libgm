#ifndef LIBGM_GRID_GRAPH_HPP
#define LIBGM_GRID_GRAPH_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/vertex_traits.hpp>

namespace libgm {

  /**
   * Specialization of vertex_traits for std::pair<int, int>.
   */
  template <>
  struct vertex_traits<std::pair<int, int> > {
    //! Returns a null pair (-1, -1).
    static std::pair<int, int> null() { return {-1, -1}; }

    //! Returns a deleted pair (-2, -2).
    static std::pair<int, int> deleted() { return {-2, -2}; }

    //! Prints the pair to an output stream.
    static void print(std::ostream& out, std::pair<int, int> v) {
      if (v == std::make_pair(-1, -1)) {
        out << "null";
      } else if (v == std::make_pair(-2, -2)) {
        out << "deleted";
      } else {
        out << '(' << v.first << ',' << v.second << ')';
      }
    }

    //! Pairs use libgm::pair_hash
    typedef pair_hash<int, int> hasher;
  };

  /**
   * Creates a graph with vertices (0, 0), ..., (m-1, n-1)
   * and connects the vertices with distance one.
   *
   * \tparam VP the vertex property type of the graph
   * \tparam EP the edge property type of the graph
   */
  template <typename VP, typename EP>
  void make_grid_graph(std::size_t m, std::size_t n,
                       undirected_graph<std::pair<int, int>, VP, EP>& g) {
    // if the grid is empty, we need to add vertex (0, 0) manually
    if (m == 1 && n == 1) {
      g.add_vertex({0, 0});
      return;
    }

    // otherwise, it's enough to add edges (vertices are added automatically)
    for (int i = 0; i < static_cast<int>(m); i++) {
      for (int j = 0; j < static_cast<int>(n); j++) {
        if (j+1 < n) { g.add_edge({i, j}, {i, j+1}); }
        if (i+1 < m) { g.add_edge({i, j}, {i+1, j}); }
      }
    }
  }

} // namespace libgm

#endif
