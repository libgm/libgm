#pragma once

#include <boost/functional/hash.hpp>

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
  /// Constructs an empty edge with null source and target.
  DirectedEdge()
    : source_(), target_() {}

  /// Constructs a special "root" edge with empty source and given target.
  explicit DirectedEdge(Vertex target)
    : source_(), target_(target) { }

  /// Constructor the standard edge.
  DirectedEdge(Vertex source, Vertex target)
    : source_(source), target_(target) { }

  /// Conversion to bool indicating if this edge is empty.
  explicit operator bool() const {
    return source_ != Vertex() && target_ != Vertex();
  }

  /// Returns the source vertex.
  Vertex source() const {
    return source_;
  }

  /// Returns the target vertex.
  Vertex target() const {
    return target_;
  }

  /// Returns the pair consisting of source and target vertex.
  std::pair<Vertex, Vertex> pair() const {
    return { source_, target_ };
  }

  /// Returns the pair cosisting of target and source vertex.
  std::pair<Vertex, Vertex> reverse_pair() const {
    return { target_, source_ };
  }

  /// Returns true if two edges have the same source and target.
  friend bool operator==(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() == b.pair();
  }

  /// Returns true if two edges do not have the same source or target.
  friend bool operator!=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() != b.pair();
  }

  /// Compares two undirected edges.
  friend bool operator<=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() <= b.pair();
  }

  /// Compares two undirected edges.
  friend bool operator>=(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() >= b.pair();
  }

  /// Compares two undirected edges.
  friend bool operator<(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() < b.pair();
  }

  /// Compares two undirected edges.
  friend bool operator>(const DirectedEdge& a, const DirectedEdge& b) {
    return a.pair() > b.pair();
  }

  /// Prints the edge to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const DirectedEdge& e) {
    out << e.source() << " --> " << e.target();
    return out;
  }

private:
  /// Vertex from which the edge originates.
  Vertex source_;

  /// Vertex to which the edge emanates.
  Vertex target_;

}; // class DirectedEdge

} // namespace libgm


namespace std {

/// \relates DirectedEdge
template <typename Vertex>
struct hash<libgm::DirectedEdge<Vertex>> {
  size_t operator()(const libgm::DirectedEdge<Vertex>& e) const {
    size_t seed = 0;
    boost::hash_combine(seed, e.source());
    boost::hash_combine(seed, e.target());
    return seed;
  }
};

} // namespace std
