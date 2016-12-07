#ifndef LIBGM_GRID_EDGE_HPP
#define LIBGM_GRID_EDGE_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/grid_vertex.hpp>

#include <iosfwd>
#include <utility>

namespace libgm {
  template <typename Index>
  class grid_edge {
  public:
    //! Constructor setting the source and target.
    grid_edge(grid_vertex<Index> source, grid_vertex<Index> target)
      : source_(source), target_(target) { }

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

    //! Compares two undirected edges.
    friend bool operator<(const grid_edge& a, const grid_edge& b) {
      return a.pair() < b.pair();
    }

    //! Prints the edge to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const gid_edge& e) {
      out << e.source() << " -- " << e.target();
      return out;
    }

  private:
    Vertex source_;
    Vertex target_;
  }; // class grid_edge

} // namespace libgm


namespace std {

  //! \relates undirected_edge
  template <typename Index>
  struct hash<libgm::grid_edge<Index>>
    : libgm::hash_pair<libgm::grid_edge<Index>> { };

} // namespace std

#endif

#endif
