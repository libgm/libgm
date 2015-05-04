#ifndef LIBGM_DIRECTED_EDGE_HPP
#define LIBGM_DIRECTED_EDGE_HPP

#include <libgm/global.hpp>
#include <libgm/functional/hash.hpp>

#include <iosfwd>
#include <utility>

namespace libgm {

  // Forward declarations
  template<typename V, typename VP, typename EP> 
  class directed_graph;

  template<typename V, typename VP, typename EP> 
  class directed_multigraph;

  /**
   * A class that represents a directed edge.
   * \ingroup graph_types
   */
  template <typename Vertex>
  class directed_edge {
  public:
    //! Constructs an empty edge with null source and target.
    directed_edge()
      : source_(), target_(), property_() {}

    //! Constructs a special "root" edge with empty source and given target.
    explicit directed_edge(Vertex target)
      : source_(), target_(target), property_() { }
    
    //! Conversion to bool indicating if this edge is empty.
    explicit operator bool() const {
      return source_ != Vertex() || target_ != Vertex();
    }

    //! Returns the pair consisting of source and target vertex.
    std::pair<Vertex, Vertex> pair() const {
      return { source_, target_ };
    }

    //! Returns the pair cosisting of target and source vertex.
    std::pair<Vertex, Vertex> reverse_pair() const {
      return { target_, source_ };
    }

    //! Compares two undirected edges.
    friend bool operator<(const directed_edge& a, const directed_edge& b) {
      return (a.source_ < b.source_) || 
        (a.source_ == b.source_ && a.target_ < b.target_);
    }

    //! Returns true if two undirected edges have the same endpoints.
    friend bool operator==(const directed_edge& a, const directed_edge& b) {
      return a.source_ == b.source_ && a.target_ == b.target_;
    }

    //! Returns true if two undirected edges do not have the same endpoints.
    friend bool operator!=(const directed_edge& a, const directed_edge& b) {
      return !(a == b);
    }

    //! Returns the source vertex.
    const Vertex& source() const {
      return source_;
    }

    //! Returns the target vertex.
    const Vertex& target() const {
      return target_;
    }

    //! Prints the edge to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const directed_edge& e) {
      out << e.source() << " --> " << e.target();
      return out;
    }

  private:
    //! Constructor which permits setting the edge_property
    directed_edge(Vertex source,
                  Vertex target,
                  void* property = NULL)
      : source_(source), target_(target), property_(property) { }

    //! Vertex from which the edge originates
    Vertex source_;

    //! Vertex to which the edge emenates
    Vertex target_;

    /**
     * The property associated with this edge. Edges maintain a private
     * pointer to the associated property.  However, this pointer can only
     * be accessed through the associated graph. This permits graphs to
     * return iterators over edges and permits constant time lookup for
     * the corresponding edge properties. The property is stored as a void*,
     * to simplify the type of the edges.
     */ 
    void* property_;

    //! Gives access to constructor and the property pointer.
    template <typename V, typename VP, typename EP>
    friend class directed_graph;

    //! Gives access to constructor and the property pointer.
    template <typename V, typename VP, typename EP>
    friend class directed_multigraph;

  }; // class directed_edge

} // namespace libgm


namespace std {

  //! \relates directed_edge
  template <typename Vertex>
  struct hash<libgm::directed_edge<Vertex>> {
    typedef libgm::directed_edge<Vertex> argument_type;
    typedef size_t result_type;
    size_t operator()(const libgm::directed_edge<Vertex>& e) const {
      size_t seed = 0;
      libgm::hash_combine(seed, e.source());
      libgm::hash_combine(seed, e.target());
      return seed;
    }
  };
  
} // namespace std

#endif
