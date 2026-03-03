#pragma once

#include <iosfwd>

namespace libgm {

/**
 * An adapter that allows bidirectional information in an undirected graph.
 * @tparam T the information stored in either direction.
 *         Must be DefaultConstructible and CopyConstructible.
 *
 * \ingroup graph_types
 */
template <typename T>
struct Bidirectional {
  T forward; ///< The T in the direction min(u,v) --> max(u,v)
  T reverse; ///< The T in the direciton max(u,v) --> min(u,v)

  Bidirectional() { }

  /// Returns the T for the directed edge (u, v).
  template <typename Vertex>
  T& operator()(Vertex u, Vertex v) {
    return u < v ? forward : reverse;
  }

  /// Returns the T for the directed edge (u, v)
  template <typename Vertex>
  const T& operator()(Vertex u, Vertex v) const {
    return u < v ? forward : reverse;
  }

  /// Returns the T for the directed edge (e.source(), e.target())
  template <typename Edge>
  T& operator()(Edge e) {
    return e.source() < e.target() ? forward : reverse;
  }

  /// Returns the T for the directed edge (e.source(), e.target())
  template <typename Edge>
  const T& operator()(Edge e) const {
    return e.source() < e.target() ? forward : reverse;
  }

}; // class Bidirectional

/// \relates Bidirectional
template <typename T>
std::ostream& operator<<(std::ostream& out, const Bidirectional<T>& p){
  out << "(" << p.forward << "," << p.reverse << ")";
  return out;
}

} // namespace libgm
