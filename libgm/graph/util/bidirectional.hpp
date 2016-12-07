#ifndef LIBGM_BIDIRECTIONAL_HPP
#define LIBGM_BIDIRECTIONAL_HPP

#include <libgm/graph/undirected_edge.hpp>

#include <iosfwd>

namespace libgm {

  /**
   * An adapter that allows bidirectional information in an undirected graph.
   * @tparam Property the information stored in either direction.
   *         Must be DefaultConstructible and CopyConstructible.
   *
   * \ingroup graph_types
   */
  template <typename Property>
  struct bidirectional {
    Property forward; //!< The property in the direction min(u,v) --> max(u,v)
    Property reverse; //!< The property in the direciton max(u,v) --> min(u,v)

    bidirectional() { }

    //! Returns the property for the directed edge (u, v).
    template <typename Vertex>
    Property& operator()(Vertex u, Vertex v) {
      return u < v ? forward : reverse;
    }

    //! Returns the property for the directed edge (u, v)
    template <typename Vertex>
    const Property& operator()(Vertex u, Vertex v) const {
      return u < v ? forward : reverse;
    }

    //! Returns the property for the directed edge (e.source(), e.target())
    template <typename Vertex>
    Property& operator()(const undirected_edge<Vertex>& e) {
      return e.source() < e.target() ? forward : reverse;
    }

    //! Returns the property for the directed edge (e.source(), e.target())
    template <typename Vertex>
    const Property& operator()(const undirected_edge<Vertex>& e) const {
      return e.source() < e.target() ? forward : reverse;
    }

  }; // class bidirectional

  //! \relates bidirectional
  template <typename Property>
  std::ostream& operator<<(std::ostream& out, const bidirectional<Property>& p){
    out << "(" << p.forward << "," << p.reverse << ")";
    return out;
  }

}

#endif
