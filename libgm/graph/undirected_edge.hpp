#ifndef LIBGM_UNDIRECTED_EDGE_HPP
#define LIBGM_UNDIRECTED_EDGE_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>

#include <algorithm>
#include <iosfwd>
#include <utility>

namespace libgm {

  /**
   * An edge of an undirected gaph, represented as the source and target
   * vertex, as well as the property pointer invisible to the caller.
   * Two undirected edges are equal only if they have the same source
   * and target vertex, even if they represent the same physical edge.
   *
   * To convert an undirected edge to an ordered pair (source, vertex),
   * use the pair() function. To convert an undirected edge to an
   * unordered pair that always places the smaller vertex first,
   * use the undordered_pair() function.
   *
   * \ingroup graph_types
   */
  template <typename Vertex>
  class undirected_edge {
  public:
    //! Constructs an empty edge with null source and target.
    undirected_edge()
      : source_(), target_(), property_() { }

    //! Construct for a special "root" edge with empty source and given target.
    explicit undirected_edge(Vertex target)
      : source_(), target_(target), property_() { }

    //! Constructor setting the source and the edge property
    undirected_edge(Vertex source, Vertex target, void* property = nullptr)
      : source_(source), target_(target), property_(property) { }

    //! Conversion to bool indicating if this edge is empty.
    explicit operator bool() const {
      return source_ != Vertex() || target_ != Vertex();
    }

    //! Returns the source vertex.
    Vertex source() const {
      return source_;
    }

    //! Returns the target vertex.
    Vertex target() const {
      return target_;
    }

    //! Returns a copy of this edge with the endpoints reversed.
    undirected_edge reverse() const {
      return undirected_edge(target_, source_, property_);
    }

    //! Returns the pair consisting of source and target vertex.
    std::pair<Vertex, Vertex> pair() const {
      return { source_, target_ };
    }

    //! Returns the pair with source and target vertex ordered
    //! s.t. first <= second.
    std::pair<Vertex, Vertex> unordered_pair() const {
      return std::minmax(source_, target_);
    }

    //! Returns true if two edges have the same source and target.
    friend bool operator==(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() == b.pair();
    }

    //! Returns true if two edges do not have the same source or target.
    friend bool operator!=(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() != b.pair();
    }

    //! Compares two undirected edges.
    friend bool operator<=(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() <= b.pair();
    }

    //! Compares two undirected edges.
    friend bool operator>=(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() >= b.pair();
    }

    //! Compares two undirected edges.
    friend bool operator<(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() < b.pair();
    }

    //! Compares two undirected edges.
    friend bool operator>(const undirected_edge& a, const undirected_edge& b) {
      return a.pair() > b.pair();
    }

    //! Prints the edge to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const undirected_edge& e) {
      out << e.source() << " -- " << e.target();
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
    void* property_;

  }; // class undirected_edge

} // namespace libgm


namespace std {

  //! \relates undirected_edge
  template <typename Vertex>
  struct hash<libgm::undirected_edge<Vertex>>
    : libgm::hash_pair<libgm::undirected_edge<Vertex>> { };

} // namespace std

#endif
