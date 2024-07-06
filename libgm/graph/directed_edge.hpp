#pragma once

#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>

#include <iosfwd>
#include <utility>

namespace libgm {

/**
 * An edge of a directed gaph, represented as the source and target vertex,
 * as well as the property pointer invisible to the caller.
 *
 * \ingroup graph_types
 */
template <typename Vertex>
class DirectedEdge {
public:
  //! Constructs an empty edge with null source and target.
  DirectedEdge()
    : source_(), target_(), property_() {}

  //! Constructs a special "root" edge with empty source and given target.
  explicit DirectedEdge(Vertex target)
    : source_(), target_(target), property_() { }

  //! Constructor which permits setting the edge_property.
  DirectedEdge(Vertex source, Vertex target, void* property = nullptr)
    : source_(source), target_(target), property_(property) { }

  //! Conversion to bool indicating if this edge is empty.
  explicit operator bool() const {
    return source_ != Vertex() && target_ != Vertex();
  }

  //! Returns the source vertex.
  Vertex source() const {
    return source_;
  }

  //! Returns the target vertex.
  Vertex target() const {
    return target_;
  }

  //! Returns the pair consisting of source and target vertex.
  std::pair<Vertex, Vertex> pair() const {
    return { source_, target_ };
  }

  //! Returns the pair cosisting of target and source vertex.
  std::pair<Vertex, Vertex> reverse_pair() const {
    return { target_, source_ };
  }

  //! Returns true if two edges have the same source and target.
  friend bool operator==(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() == b.pair();
  }

  //! Returns true if two edges do not have the same source or target.
  friend bool operator!=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() != b.pair();
  }

  //! Compares two undirected edges.
  friend bool operator<=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() <= b.pair();
  }

  //! Compares two undirected edges.
  friend bool operator>=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() >= b.pair();
  }

  //! Compares two undirected edges.
  friend bool operator<(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() < b.pair();
  }

  //! Compares two undirected edges.
  friend bool operator>(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() > b.pair();
  }

  //! Prints the edge to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const DirectedEdge& e) {
    out << e.source() << " --> " << e.target();
    return out;
  }

private:
  //! Vertex from which the edge originates
  Vertex source_;

  //! Vertex to which the edge emenates
  Vertex target_;

  /**
   * The property associated with this edge. Edges maintain a private
   * pointer to the associated property, which permits a constant-time
   * lookup of edge properties. Because edge properties are only accessed
   * through the graph, we can type-erase the property type here and
   * store the property as void*.
   */
  Object* property_;

}; // class DirectedEdge

} // namespace libgm


namespace std {

  //! \relates DirectedEdge
  template <typename Vertex>
  struct hash<libgm::DirectedEdge<Vertex>>
    : libgm::hash_pair<libgm::DirectedEdge<Vertex>> { };

} // namespace std
