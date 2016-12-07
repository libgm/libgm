#ifndef LIBGM_BIPARTITE_EDGE_HPP
#define LIBGM_BIPARTITE_EDGE_HPP

#include <iosfwd>

namespace libgm {

  // TODO: hashing

  /**
   * An edge of a bipartite graph, represented as source and target vertex,
   * directionality, and edge property pointer invisible to the caller.
   * Two bipartite edges are equal only if they have the same source and
   * target vertex and directionality.
   */
  template <typename Vertex1, typename Vertex2>
  class bipartite_edge {
  public:
    //! Creates an empty edge with null soruce and target.
    bipartite_edge()
      : v1_(),  v2_(), forward_(true), property_() { }

    //! Constructs a forward edge
    bipartite_edge(Vertex1 v1, Vertex2 v2, void* property = nullptr)
      : v1_(v1), v2_(v2), forward_(true), property_(property) { }

    //! Constructs a backward edge
    bipartite_edge(Vertex2 v2, Vertex1 v1, void* property = nullptr)
      : v1_(v1), v2_(v2), forward_(false), property_(property) { }

    //! Conversion to bool indicating if this edge is empty.
    explicit operator bool() const {
      return v1_ != Vertex1() || v2_ != Vertex2();
    }

    //! Returns the type-1 endpoint of this edge.
    Vertex1 v1() const {
      return v1_;
    }

    //! Returns the type-2 endpoing of this edge.
    Vertex2 v2() const {
      return v2_;
    }

    //! Returns true if the edge goes from a type-1 vertex to a type-2 vertex.
    bool forward() const {
      return forward_;
    }

    //! Returns a copy of this edge with endpoints reversed.
    bipartite_edge reverse() const {
      bipartite_edge e = *this;
      e.forward_ = !e.forward_;
      return e;
    }

    //! Returns the endpoints of this edge.
    std::pair<Vertex1, Vertex2> endpoints() const {
      return { v1_, v2_ };
    }

    //! Returns the endpoints of this edge and directionality as a tuple.
    std::tuple<Vertex1, Vertex2, bool> tuple() const {
      return { v1_, v2_, forward_ };
    }

    //! Returns true if two edges have the same source and target.
    friend bool operator==(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() == b.tuple();
    }

    //! Returns true if two edges do not have the same source or target.
    friend bool operator!=(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() != b.tuple();
    }

    //! Compares two bipartite edges.
    friend bool operator<=(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() <= b.tuple();
    }

    //! Compares two bipartite edges.
    friend bool operator>=(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() >= b.tuple();
    }

    //! Compares two bipartite edges.
    friend bool operator<(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() < b.tuple();
    }

    //! Compares two bipartite edges.
    friend bool operator>(const bipartite_edge& a, const bipartite_edge& b) {
      return a.tuple() > b.tuple();
    }

    //! Prints the edge to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const bipartite_edge& e) {
      if (e.forward_) {
        out << e.v1_ << " -- " << e.v2_;
      } else {
        out << e.v2_ << " -- " << e.v1_;
      }
      return out;
    }

  private:
    Vertex1 v1_;
    Vertex2 v2_;
    bool forward_; // true if the edge is from type1 to type2
    void* property_;

  }; // class bipartite_edge

} // namespace libgm

#endif
