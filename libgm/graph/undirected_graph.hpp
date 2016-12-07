#ifndef LIBGM_UNDIRECTED_GRAPH_HPP
#define LIBGM_UNDIRECTED_GRAPH_HPP

#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/util/vertex_edge_property_iterator.hpp>
#include <libgm/graph/util/void.hpp>
#include <libgm/iterator/map_bind1_iterator.hpp>
#include <libgm/iterator/map_bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <boost/graph/graph_traits.hpp>

#include <iterator>
#include <iosfwd>
#include <unordered_map>

namespace libgm {

  /**
   * A class that representeds an undirected graph as an adjancy list (map).
   * The template is paraemterized by the vertex type as well as the type
   * of properties associated with vertices and edges.
   *
   * \tparam Vertex
   *         The type that represents a vertex.
   * \tparam VertexProperty
   *         The type of data associated with vertices.
   * \tparam EdgeProperty
   *         The type of data associated with edges.
   *
   * \ingroup graph_types
   */
  template <typename Vertex,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class undirected_graph {

    // Private types
    //--------------------------------------------------------------------------
  private:
    //! The map type used to associate neighbors and edge data with each vertex.
    using neighbor_map = std::unordered_map<Vertex, EdgeProperty*>;

    /**
     * A struct with the data associated with each vertex. This structure
     * stores the property associated with the vertex as well all edges from/to
     * parent and child vertices (along with the edge properties).
     */
    struct vertex_data {
      VertexProperty property;
      neighbor_map neighbors;
      vertex_data() : property() { }
    };

    //! The map types that associates all the vertices with their vertex_data.
    using vertex_data_map = std::unordered_map<Vertex, vertex_data>;

    // Public type declerations
    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge type, and properties
    using vertex_type     = Vertex;
    using edge_type       = undirected_edge<Vertex>;
    using vertex_property = VertexProperty;
    using edge_property   = EdgeProperty;

    // Iterators
    using vertex_iterator   = map_key_iterator<vertex_data_map>;
    using neighbor_iterator = map_key_iterator<neighbor_map>;
    using in_edge_iterator  = map_bind2_iterator<neighbor_map, edge_type>;
    using out_edge_iterator = map_bind1_iterator<neighbor_map, edge_type>;
    class edge_iterator;

    // Constructors and destructors
    //--------------------------------------------------------------------------
  public:
    //! Create an empty graph.
    undirected_graph()
      : num_edges_(0) { }

    //! Create a graph from a list of pairs of vertices.
    template <typename Range>
    explicit undirected_graph(const Range& edges, typename Range::iterator* = 0)
      : num_edges_(0) {
      for (std::pair<Vertex, Vertex> vp : edges) {
        add_edge(vp.first, vp.second);
      }
    }

    //! copy constructor.
    undirected_graph(const undirected_graph& g) {
      *this = g;
    }

    //! Destructor.
    ~undirected_graph() {
      free_edge_data();
    }

    //! Assignment operator.
    undirected_graph& operator=(const undirected_graph& g) {
      if (this == &g) return *this;
      free_edge_data();
      data_ = g.data_;
      num_edges_ = g.num_edges_;
      for (edge_type e : edges()) {
        if(e.property_ != nullptr) {
          EdgeProperty* ptr =
            new EdgeProperty(*static_cast<edge_property*>(e.property_));
          data_[e.source()].neighbors[e.target()] = ptr;
          data_[e.target()].neighbors[e.source()] = ptr;
        }
      }
      return *this;
    }

    //! Swaps two graphs in constant time.
    friend void swap(undirected_graph& a, undirected_graph& b) {
      swap(a.data_, b.data_);
      std::swap(a.num_edges_, b.num_edges_);
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return { data_.begin(), data_.end() };
    }

    //! Returns the range of all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { { data_.begin(), data_.end() }, { data_.end(), data_.end() } };
    }

    //! Returns the vertices adjacent to u.
    iterator_range<neighbor_iterator>
    neighbors(Vertex u) const {
      const vertex_data& data(data_.at(u));
      return { data.neighbors.begin(), data.neighbors.end() };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(Vertex u) const {
      const vertex_data& data(data_.at(u));
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns the edges outgoing from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(Vertex u) const {
      const vertex_data& data(data_.at(u));
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(Vertex u) const {
      return data_.find(u) != data_.end();
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(Vertex u, Vertex v) const {
      auto it = data_.find(u);
      return it != data_.end() && it->second.neighbors.count(v);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(undirected_edge<Vertex> e) const {
      return contains(e.source(), e.target());
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    undirected_edge<Vertex> edge(Vertex u,  Vertex v) const {
      return { u, v, data_.at(u).neighbors.at(v) };
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t in_degree(Vertex u) const {
      return data_.at(u).neighbors.size();
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t out_degree(Vertex u) const {
      return data_.at(u).neighbors.size();
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t degree(Vertex u) const {
      return data_.at(u).neighbors.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data_.size() == 0;
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return data_.size();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      return num_edges_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    undirected_edge<Vertex> reverse(undirected_edge<Vertex> e) const {
      return e.reverse();
    }

    //! Returns the property associated with a vertex,
    const VertexProperty& operator[](Vertex u) const {
      return data_.at(u).property;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](Vertex u) {
      return data_.at(u).property;
    }

    //! Returns the property associated with an edge.
    const EdgeProperty& operator[](undirected_edge<Vertex> e) const {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](undirected_edge<Vertex> e) {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge must exist.
     */
    const EdgeProperty& operator()(Vertex u, Vertex v) const {
      return *static_cast<EdgeProperty*>(edge(u, v).property_);
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge is added if necessary.
     */
    EdgeProperty& operator()(Vertex u, Vertex v) {
      return *static_cast<EdgeProperty*>(add_edge(u, v).first.property_);
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
     * Compares the graph strucutre and the vertex & edge properties.
     * The property types must support operator!=().
     */
    bool operator==(const undirected_graph& other) const {
      if (num_vertices() != other.num_vertices() ||
          num_edges() != other.num_edges()) {
        return false;
      }
      for (const auto& vp : data_) {
        auto vit = other.data_.find(vp.first);
        if (vit == other.data_.end() ||
            vp.second.property != vit->second.property) {
          return false;
        }
        const neighbor_map& neighbors = vit->second.neighbors;
        for (const auto& ep : vp.second.neighbors) {
          auto eit = neighbors.find(ep.first);
          if (eit == neighbors.end() || *ep.second != *eit->second) {
            return false;
          }
        }
      }
      return true;
    }

    //! Inequality comparison
    bool operator!=(const undirected_graph& other) const {
      return !(*this == other);
    }

    //! Prints the graph to an output stream
    friend std::ostream&
    operator<<(std::ostream& out, const undirected_graph& g) {
      out << "Vertices" << std::endl;
      for (Vertex v : g.vertices()) {
        out << v << ": " << g[v] << std::endl;
      }
      out << "Edges" << std::endl;
      for (undirected_edge<Vertex> e : g.edges()) {
        out << e << std::endl;
      }
      return out;
    }

    // Modifications
    //--------------------------------------------------------------------------

    /**
     * Adds a vertex to a graph and associate the property with that vertex.
     * If the vertex is already present, its property is not overwritten.
     * \returns true if the insertion took place (i.e., vertex was not present).
     */
    bool add_vertex(Vertex u, const VertexProperty& p = VertexProperty()) {
      assert(u != null_vertex());
      if (contains(u)) {
        return false;
      } else {
        data_[u].property = p;
        return true;
      }
    }

    /**
     * Adds an edge {u,v} to teh graph. If the edge already exists, its
     * property is not overwritten. If u and v are present, they are added.
     * \return the edge and bool indicating whether the edge was newly added.
     */
    std::pair<undirected_edge<Vertex>, bool>
    add_edge(Vertex u, Vertex v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != null_vertex());
      assert(v != null_vertex());
      auto uit = data_.find(u);
      if (uit != data_.end()) {
        auto vit = uit->second.neighbors.find(v);
        if (vit != uit->second.neighbors.end()) {
          return std::make_pair(edge_type(u, v, vit->second), false);
        }
      }
      EdgeProperty* ptr = new EdgeProperty(p);
      data_[u].neighbors[v] = ptr;
      data_[v].neighbors[u] = ptr;
      ++num_edges_;
      return std::make_pair(edge_type(u, v, ptr), true);
    }

    //! Adds edges among all vertices in an undirected graph.
    template <typename Range>
    void add_clique(const Range& vertices) {
      auto it1 = vertex.begin(), end = vertices.end();
      for (; it1 != end; ++it1) {
        graph.add_vertex(*it1);
        for (auto it2 = std::next(it); it2 != end; ++it2) {
          graph.add_edge(*it1, *it2);
        }
      }
    }

    //! Removes a vertex from the graph and all its incident edges.
    void remove_vertex(Vertex u) {
      remove_edges(u);
      data_.erase(u);
    }

    //! Removes an undirected edge {u, v}.
    void remove_edge(Vertex u, Vertex v) {
      vertex_data& data = data_.at(u);
      auto it = data.neighbors.find(v);
      assert(it != data.neighbors.end());

      // delete the edge data and the edge itself
      delete it->second;
      data.neighbors.erase(it);
      data_[v].neighbors.erase(u);
      --num_edges_;
    }

    //! Removes all edges incident to a vertex.
    void remove_edges(Vertex u) {
      neighbor_map& neighbors = data_.at(u).neighbors;
      num_edges_ -= neighbors.size();
      for (const auto& n : neighbors) {
        if(n.first != u) { data_[n.first].neighbors.erase(u); }
        if(n.second != nullptr) { delete n.second; }
      }
      neighbors.clear();
    }

    //! Removes all edges from the graph.
    void remove_edges() {
      free_edge_data();
      for (auto& p : data_) {
        p.second.neighbors.clear();
      }
      num_edges_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data_.clear();
      num_edges_ = 0;
    }

    //! Saves the graph to an archive
    void save(oarchive& ar) const {
      ar << num_vertices();
      ar << num_edges();
      for (const auto& p : data_) {
        ar << p.first << p.second.property;
      }
      for (edge_type e : edges()) {
        ar << e.source() << e.target() << operator[](e);
      }
    }

    //! Loads the graph from an archive
    void load(iarchive& ar) {
      clear();
      std::size_t num_vertices, num_edges;
      Vertex u, v;
      ar >> num_vertices;
      ar >> num_edges;
      while (num_vertices-- > 0) {
        ar >> v;
        ar >> data_[v].property;
      }
      while (num_edges -- > 0) {
        ar >> u >> v;
        ar >> operator()(u, v);
      }
    }

    // Implementation of edge iterator
    //--------------------------------------------------------------------------
  public:
    class edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      using reference = edge_type;
      using outer_iterator = typename vertex_data_map::const_iterator;
      using inner_iterator = typename neighbor_map::const_iterator;

      edge_iterator() {}

      edge_iterator(outer_iterator it1, outer_iterator end1)
        : it1_(it1), end1_(end1) {
        find_next();
      }

      edge_type operator*() const {
        return edge_type(it1_->first, it2_->first, it2_->second);
      }

      edge_iterator& operator++() {
        do {
          ++it2_;
        } while (it2_ != it1_->second.neighbors.end() &&
                 it2_->first < it1_->first);
        if (it2_ == it1_->second.neighbors.end()) {
          ++it1_;
          find_next();
        }
        return *this;
      }

      edge_iterator operator++(int) {
        edge_iterator copy = *this;
        operator++();
        return copy;
      }

      bool operator==(const edge_iterator& o) const {
        return
          (it1_ == end1_ && o.it1_ == o.end1_) ||
          (it1_ == o.it1_ && it2_ == o.it2_);
      }

      bool operator!=(const edge_iterator& other) const {
        return !(operator==(other));
      }

    private:
      //! find the next non-empty neighbor map with it1_->firstt <= it2_->first
      void find_next() {
        while (it1_ != end1_) {
          it2_ = it1_->second.neighbors.begin();
          while (it2_ != it1_->second.neighbors.end() &&
                 it2_->first < it1_->first) {
            ++it2_;
          }
          if (it2_ != it1_->second.neighbors.end()) {
            break;
          } else {
            ++it1_;
          }
        }
      }

      outer_iterator it1_;  //!< the iterator to the vertex data
      outer_iterator end1_; //!< the iterator past the last vertex data
      inner_iterator it2_;  //!< the iterator to the current neighbor

    }; // class edge_iterator

    // Private members
    //--------------------------------------------------------------------------
  private:
    void free_edge_data() {
      for (edge_type e : edges()) {
        if(e.property_) {
          delete static_cast<EdgeProperty*>(e.property_);
        }
      }
    }

    //! A map from each vertex to its vertex data.
    vertex_data_map data_;

    //! The total number of edges in the graph.
    std::size_t num_edges_;

  }; // class undirected_graph

} // namespace libgm


namespace boost {

  //! Type declarations that let our graph structure work in BGL algorithms
  template <typename Vertex, typename VP, typename EP>
  struct graph_traits< libgm::undirected_graph<Vertex, VP, EP> > {

    typedef libgm::undirected_graph<Vertex, VP, EP> graph_type;

    typedef typename graph_type::vertex_type        vertex_descriptor;
    typedef typename graph_type::edge_type          edge_descriptor;
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  adjacency_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;

    typedef undirected_tag                          directed_category;
    typedef disallow_parallel_edge_tag              edge_parallel_category;

    struct traversal_category :
      public virtual boost::vertex_list_graph_tag,
      public virtual boost::incidence_graph_tag,
      public virtual boost::adjacency_graph_tag,
      public virtual boost::edge_list_graph_tag { };

    typedef std::size_t vertices_size_type;
    typedef std::size_t edges_size_type;
    typedef std::size_t degree_size_type;

    static vertex_descriptor null_vertex() {
      return Vertex();
    }

  };

} // namespace boost

#endif
