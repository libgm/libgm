#ifndef LIBGM_UNDIRECTED_EDGE_HPP
#define LIBGM_UNDIRECTED_EDGE_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/vertex_traits.hpp>

#include <algorithm>
#include <iosfwd>
#include <utility>

namespace libgm {

  // Forward declaration
  template<typename V, typename VP, typename EP>
  class undirected_graph;

  /**
   * A class that represents an undirected edge.
   * An undirected edge is represented by a source and target vertex
   * (as well as the cached edge property pointer invisible to the caller).
   * This allows us to differentiate incoming vs. outgoing edges in
   * undirected graphs: the outoing edges of a vertex will always have that
   * vertex as a source, whereas the incoing edges of a vertex will always
   * have that vertex as a target. Nevertheless, for the purpose of
   * comparisons and hashing, two undirected edges (u, v) and (v, u)
   * are considered equivalent.
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
      : source_(vertex_traits<Vertex>::null()),
        target_(vertex_traits<Vertex>::null()),
        property_() { }

    //! Construct for a special "root" edge with empty source and given target.
    explicit undirected_edge(Vertex target)
      : source_(vertex_traits<Vertex>::null()),
        target_(target),
        property_() { }

    //! Conversion to bool indicating if this edge is empty.
    explicit operator bool() const {
      return source_ != vertex_traits<Vertex>::null() ||
             target_ != vertex_traits<Vertex>::null();
    }

    //! Returns the source vertex.
    const Vertex& source() const {
      return source_;
    }

    //! Returns the target vertex.
    const Vertex& target() const {
      return target_;
    }

    //! Returns the edge with the endpoints reversed.
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

    //! Compares two undirected edges.
    friend bool operator<(const undirected_edge& a, const undirected_edge& b) {
      return a.unordered_pair() < b.unordered_pair();
    }

    //! Returns true if two undirected edges have the same endpoints.
    friend bool operator==(const undirected_edge& a, const undirected_edge& b) {
      return a.unordered_pair() == b.unordered_pair();
    }

    //! Returns true if two undirected edges do not have the same endpoints.
    friend bool operator!=(const undirected_edge& a, const undirected_edge& b) {
      return a.unordered_pair() != b.unordered_pair();
    }

    //! Returns the hash value of an undirected edge.
    friend std::size_t hash_value(const undirected_edge& e) {
      std::size_t seed = 0;
      std::pair<Vertex, Vertex> p = std::minmax(e.source(), e.target());
      libgm::hash_combine(seed, p.first);
      libgm::hash_combine(seed, p.second);
      return seed;
    }

    //! Prints the edge to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const undirected_edge& e) {
      vertex_traits<Vertex>::print(out, e.source());
      out << " -- ";
      vertex_traits<Vertex>::print(out, e.target());
      return out;
    }

  private:
    //! Constructor setting the source and the edge property
    undirected_edge(const Vertex& source,
                    const Vertex& target,
                    void* property)
      : source_(source), target_(target), property_(property) { }

    //! Vertex from which the edge originates
    Vertex source_;

    //! Vertex to which the edge emenates
    Vertex target_;

    /**
     * The property associated with this edge.  Edges maintain a private
     * pointer to the associated property.  However, this pointer can only
     * be accessed through the associated graph. This permits graphs to
     * return iterators over edges and permits constant time lookup for
     * the corresponding edge properties. The property is stored as a void*,
     * to simplify the type of the edges.
     */
    void* property_;

    //! Gives access to constructor and the property pointer.
    template <typename V, typename VP, typename EP>
    friend class undirected_graph;

  }; // class undirected_edge

} // namespace libgm


namespace std {

  //! \relates undirected_edge
  template <typename Vertex>
  struct hash<libgm::undirected_edge<Vertex>>
    : libgm::default_hash<libgm::undirected_edge<Vertex>> { };

} // namespace std

#endif
