#ifndef LIBGM_GRID_GRAPH_HPP
#define LIBGM_GRID_GRAPH_HPP

#include <libgm/functional/hash.hpp>
#include <libgm/graph/grid_vertex.hpp>
#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/vertex_traits.hpp>

#include <vector>

namespace libgm {

  /**
   * An undirected graph, whose vertices form an integral grid.
   * The graph vertices and edges are associated with properties.
   *
   * \tparam Index
   *         The type representing the grid index. Must be a signed integer.
   * \tparam VertexProperty
   *         The type representing the property associated with each vertex.
   *         Must be DefaultConstructible and CopyConstructible.
   * \tparam EdgeProperty
   *         The type representing the property associated with each edge.
   *         Must be DefaultConstructible and CopyConstructible.
   */
  template <typename Index = int,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class grid_graph {

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge types, and properties
    using vertex_type     = grid_vertex<Index>;
    using edge_type       = grid_edge<Index>;
    using vertex_property = VertexProperty;
    using edge_property   = EdgeProperty;

    // Iterators
    class vertex_iterator;
    class neighbor_iterator;
    class in_edge_iterator;
    class out_edge_iterator;
    class edge_iterator;

    // Constructors and destructors
    //--------------------------------------------------------------------------

    //! Creates an empty graph.
    grid_graph() { }

    //! Creates a graph with the given number of rows and columns.
    grid_graph(std::size_t rows, std::size_t cols)
      : size_(rows, cols), vp_(rows * cols), ep_(rows * cols * 2) { }

    //! Swaps two graphs in constant time.
    friend void swap(grid_graph& a, grid_graph& b) {
      using std::swap;
      swap(a.size_, b.size_);
      swap(a.vp_, b.vp_);
      swap(a.ep_, b.ep_);
    }

    //! Saves the graph to an archive
    void save(oarchive& ar) const {
      ar << size_ << vp_ << ep_;
    }

    //! Loads the graph from an archive
    void load(iarchive& ar) {
      ar >> size_ >> vp_ >> ep_;
    }

    //! Returns true if two grid graphs have the same shape and properties.
    friend bool operator==(const grid_graph& a, const grid_graph& b) {
      return a.size_ == b.size_ && a.vp_ == b.vp_ && a.ep_ == b.ep_;
    }

    //! Returne true if two grid graphs do not have same shape and properties.
    friend bool operator!=(const grid_graph& a, const grid_graph& b) {
      return !(a == b);
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      if (empty()) {
        return { };
      } else {
        return { { vertex_type(), size_ }, size_ };
      }
    }

    //! Returns the range of all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      if (empty()) {
        return { };
      } else {
        return { { vertex_type(), size_ }, size_ };
      }
    }

    //! Returns the neighbors of a vertex.
    iterator_range<neighbor_iterator>
    neighbors(grid_vertex<Index> u) const {
      return { { u, neighbor_mask(u) }, {} };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(grid_vertex<Index> u) const {
      return { { u, neighbor_mask(u) }, {} };
    }

    //! Returns the edges outgoing from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(grid_vetex<Index> u) const {
      return { { u, neighbor_mask(u) }, {} };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(grid_vertex<Index> u) const {
      return u >= vertex_type() && u < size_;
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(grid_vertex<Index> u, grid_vertex<Index> v) const {
      return contains(u) && contains(v) && abs(u - v).sum() == Index(1);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(grid_edge<Index> e) const {
      return contains(e.source(), e.target());
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    grid_edge<Index> edge(grid_vertex<Index> u, grid_vertex<Index> v) const {
      assert(contains(u, v));
      return { u, v };
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t in_degree(grid_vertex<Index> u) const {
      return degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t out_degree(grid_vertex<Index> u) const {
      return degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t degree(grid_vertex<Index> u) const {
      return count_greater(u, vertex_type()) + count_less(u, size_ - 1);
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return size_.row == 0 || size_.col == 0;
    }

    //! Returns the number of rows of the graph.
    std::size_t rows() const {
      return size_.row;
    }

    //! Returns the number of columns of the graph.
    std::size_t cols() const {
      return size_.col;
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return rows() * cols();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      if (empty()) {
        return 0;
      } else {
        return num_vertices() * 2 - size_.sum();
      }
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    grid_edge<Index> reverse(grid_edge<Index> e) const {
      return e.reverse();
    }

    //! Returns the property associated with a vertex,
    const VertexProperty& operator[](grid_vertex<Index> u) const {
      return vp_[linear(u)];
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](grid_vertex<Index> u) {
      return vp_[linear(u)];
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator()(Index r, Index c) const {
      return vp_[linear(grid_vertex<Index>(r, c))];
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator()(Index r, Index c) {
      return vp_[linear(grid_vertex<Index>(r, c))];
    }

    //! Returns the property associated with an edge.
    const EdgeProperty& operator[](grid_edge<Index> e) const {
      assert(contains(e));
      vertex_type u = e.forward() ? e.source() : e.target();
      return ep_[linear(u) * 2 + (e.source().row == e.target().row)];
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](grid_edge<Index> e) {
      assert(contains(e));
      vertex_type u = e.forward() ? e.source() : e.target();
      return ep_[linear(u) * 2 + (e.source().row == e.target().row)];
    }

    /**
     * Returns the begin iterator over the annotated properties of this graph.
     * This is only implementd when VertexProperty and EdgeProperty denote the
     * same type.
     */
    LIBGM_ENABLE_IF((std::is_same<VertexProperty, EdgeProperty>::value))
    vertex_edge_property_iterator<undirected_graph>
    begin() const {
      iterator_range<vertex_iterator> range = vertices();
      return { range.begin(), range.end(), edges().begin() };
    }

    /**
     * Returns the end iterator over the annotated properties of this graph.
     * This is only implementd when VertexProperty and EdgeProperty denote the
     * same type.
     */
    LIBGM_ENABLE_IF((std::is_same<VertexProperty, EdgeProperty>::value))
    vertex_edge_property_iterator<undirected_graph>
    end() const {
      iterator_range<vertex_iterator> range = vertices();
      return { range.end(), range.end(), edges().end() };
    }

    /**
     * Prints the graph to an output stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const grid_graph& g) {
      out << "Vertices" << std::endl;
      for (grid_vertex<Index> v : g.vertices()) {
        out << v << ": " << g[v] << std::endl;
      }
      out << "Edges" << std::endl;
      for (grid_edge<Index> e : g.edges()) {
        out << e << ": " << g[e] << std::endl;
      }
      return out;
    }

    // Implementation of iterators
    //--------------------------------------------------------------------------

    class vertex_iterator
      : public std::iterator<std::forward_iterator_tag, vertex_type> {
    public:
      using reference = vertex_type;

      vertex_iterator() { }

      vertex_iterator(vertex_type size)                // end constructor
        : u_(0, size.col), size_(size) { }

      vertex_iterator(vertex_type u, vertex_type size) // begin constructor
        : u_(u), size_(size) { }

      explicit operator bool() const {
        return u_.col != size_.col;
      }

      vertex_type operator*() const {
        return u_;
      }

      vertex_iterator& operator++() {
        ++u_.row;
        if (u_.row == size_.row) {
          u_.row = 0;
          ++u_.col;
        }
        return *this;
      }

      vertex_iterator operator++(int) {
        vertex_iterator copy = *this;
        operator++();
        return copy;
      }

      bool operator==(const vertex_iterator& other) const {
        return u_ == other.u_;
      }

      bool operator!=(const vertex_iterator& other) const {
        return u_ != other.u_;
      }

    private:
      vertex_type u_;
      vertex_type size_;
    }; // class vertex_iterator


    class edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      using reference = edge_type;

      edge_iterator()
        : vertical_(false) { }

      edge_iterator(vertex_type size)                // end constructor
        : u_(size - 1), size_(size), vertical_(false) { }

      edge_iterator(vertex_type u, vertex_type size) // begin constructor
        : u_(u), size_(size), vertical_(size.row > 1) { }

      edge_type operator*() const {
        return { u_, vertical_ ? u_.below() : u_.right() };
      }

      edge_iterator& operator++() {
        if (vertical_ && u_.col < size_.col - 1) {
          vertical_ = false; // not in the rightmost column
        } else {
          ++u_.row;
          if (u_.row == size_.row) {
            u_.row = 0;
            ++u_.col;
          }
          vertical = u_.row < size_.row - 1; // not in the bottommost row?
        }
        return *this;
      }

      edge_iterator operator++(int) {
        edge_iterator result(*this);
        operator++();
        return result;
      }

      bool operator==(const edge_iterator& other) const {
        return u_ == other.u_ && horizontal_ == u.horizontal_;
      }

      bool operator!=(const edge_iterator& other) const {
        return u_ != other.u_ || horizontal_ != u.horizontal_;
      }

    private:
      vertex_type u_;
      vertex_type size_;
      bool horizontal_;
    }; // class edge_iterator


    template <typename Derived>
    class neighbor_base {
    public:
      using iterator_category = std::forward_iterator_tag;
      using difference_type   = std::ptrdiff_t;

      // end iterator
      neighbor_base()
        : i_(4) { }

      // begin iterator
      neighbor_base(vertex_type u, int mask)
        : u_(u), mask_(mask), i_(-1) {
        operator++();
      }

      Derived& derived() {
        return static_cast<Derived&>(*this);
      }

      Derived& operator++() {
        for (++i_; i_ < 4; ++i_) {
          if (mask & (1 << i)) {
            break;
          }
        }
        return derived();
      }

      Derived operator++(int) {
        Derived copy = derived();
        operator++();
        return copy;
      }

      bool operator==(const neighbor_base& other) const {
        return i_ == other.i_;
      }

      bool operator!=(const neighbor_base& other) const {
        return i_ != other.i_;
      }

    protected:
      vertex_type origin() const {
        return u_;
      }

      vertex_type neighbor() const {
        return u_ + neighbor_map[i_];
      }

    private:
      vertex_type u_;
      int mask_;
      int i_;
    }; // class neighbor_base


    class neighbor_iterator
      : public neighbor_base<neighbor_iterator> {
    public:
      using value_type = vertex_type;
      using reference  = vertex_type;
      using pointer    = const vertex_type*;

      neighbor_iterator() { }

      neighbor_iterator(vertex_type u, int mask)
        : neighbor_base<neighbor_iterator>(u, mask) { }

      vertex_type operator*() const {
        return this->neighbor();
      }
    };


    class in_edge_iterator
      : public neighbor_base<in_edge_iterator> {
    public:
      using value_type = edge_type;
      using reference  = edge_type;
      using pointer    = const edge_type*;

      in_edge_iterator() { }

      in_edge_iterator(vertex_type u, int mask)
        : neighbor_base<in_edge_iterator>(u, mask) { }

      edge_type operator*() const {
        return { this->neighbor(), this->origin() };
      }
    };


    class out_edge_iterator
      : public neighbor_base<out_edge_iterator> {
    public:
      using value_type = edge_type;
      using reference  = edge_type;
      using pointer    = const edge_type*;

      out_edge_iterator() { }

      out_edge_iterator(vertex_type u, int mask)
        : neighbor_base<out_edge_iterator>(u, mask) { }

      edge_type operator*() const {
        return { this->origin(), this->neighbor()  };
      }
    };

  private:
    //! Returns the linear index of a vertex in the column-major order.
    std::size_t linear(grid_vertex<Index> u) const {
      assert(contains(u));
      return u.row + u.col * rows();
    }

    int neighbor_mask(grid_vertex<Index> u) const {
      return
        int(u.row > 0) |
        int(u.col > 0) << 1 |
        int(u.row < size_.row - 1) << 2 |
        int(u.col < size_.col - 1) << 3;
    }

    static grid_vertex<Index> neighbor_map[4] = {
      {-1, 0}, {0, -1}, {1, 0}, {0, 1}
    };

    grid_vertex<Index> size_;
    std::vector<VertexProperty> vp_;
    std::vector<EdgeProperty> ep_;
  };

} // namespace libgm

#endif
