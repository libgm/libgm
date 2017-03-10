#ifndef LIBGM_GRID_EDGE_HPP
#define LIBGM_GRID_EDGE_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/grid_vertex.hpp>

#include <iosfwd>
#include <utility>

namespace libgm {
  template <typename Index = int>
  class grid_edge {
  public:
    //! Constructor accepting a pair of vertices.
    grid_edge(grid_vertex<Index> source, grid_vertex<Index> target)
      : source_(source), target_(target) { }

    //! Constructor accepting a pair of row/column.
    grid_edge(Index source_row, Index source_col,
              Index target_row, Index target_col)
      : source(source_row, source_col), target(target_row, target_col) { }

    //! Returns the source vertex.
    grid_vertex<Index> source() const {
      return source_;
    }

    //! Returns the target vertex.
    grid_vertex<Index> target() const {
      return target_;
    }

    //! Returns true if this edge goes from a "smaller" to a "larger" vertex.
    bool forward() const {
      return source_ <= target_;
    }

    //! Returns a copy of this edge with the endpoints reversed.
    grid_edge reversed() const {
      return { target_, source_ } ;
    }

    //! Returns the pair consisting of source and target vertex.
    std::pair<grid_vertex<Index>, grid_vertex<Index> > pair() const {
      return { source_, target_ };
    }

    //! Returns true if two edges have the same source and target.
    friend bool operator==(const grid_edge& a, const grid_edge& b) {
      return a.pair() == b.pair();
    }

    //! Returns true if two edges do not have the same source or target.
    friend bool operator!=(const grid_edge& a, const grid_edge& b) {
      return a.pair() != b.pair();
    }

    //! Prints the edge to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const gid_edge& e) {
      out << e.source() << " -- " << e.target();
      return out;
    }

  private:
    grid_vertex<Index> source_;
    grid_vertex<Index> target_;
  }; // class grid_edge

  /**
   * Helper to construct an edge from two vertices, inferring the type.
   * \relates grid_edge
   */
  template <typename Index>
  inline grid_edge<Index>
  make_grid_edge(grid_vertex<Index> u, grid_vertex<Index> v) {
    return { u, v };
  }

} // namespace libgm


namespace std {

  //! \relates undirected_edge
  template <typename Index>
  struct hash<libgm::grid_edge<Index>>
    : libgm::hash_pair<libgm::grid_edge<Index>> { };

} // namespace std

#endif

#endif
