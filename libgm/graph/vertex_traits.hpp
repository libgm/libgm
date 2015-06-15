#ifndef LIBGM_VERTEX_TRAITS_HPP
#define LIBGM_VERTEX_TRAITS_HPP

#include <functional>
#include <iosfwd>

namespace libgm {

  /**
   * The vertex_traits struct provides a standard way to query properties
   * of vertices. Presently, it provides two functions that return special
   * vertex values. The library user can specialize this class to customize
   * its behavior for their own type.
   *
   * \tparam Vertex a vertex type whose properties this class describes
   */
  template <typename Vertex>
  struct vertex_traits {
    /**
     * Returns the null vertex. A null vertex cannot be inserted into the
     * graph structures and is used in graph algorithms to represent a
     * non-existent vertex. By default, this function simply returns
     * Vertex(), e.g., for integral vertex types, this corresponds to 0.
     */
    static Vertex null() {
      return Vertex();
    }

    /**
     * Returns the deleted vertex. A deleted vertex cannot be inserted
     * into the graph structures and is used in Google sparse and dense
     * hash map to denote deleted keys. By default, this function is
     * omitted and fails to compile with datastructures that require it.
     * The user can specialize vertex_traits if this function is needed.
     */
    static Vertex deleted() = delete;

    /**
     * Prints the specified vertex to an output stream.
     */
    static void print(std::ostream& out, Vertex v) {
      out << v;
    }

    /**
     * A type that represents the hash function for this vertex type.
     * Defaults to std::hash.
     */
    typedef std::hash<Vertex> hasher;

  }; // struct vertex_traits

} // namespace libgm

#endif
