#ifndef LIBGM_UNDIRECTED_GRAPH_HPP
#define LIBGM_UNDIRECTED_GRAPH_HPP

#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/void.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/serialize.hpp>

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
   * \tparam Vertex the type that represents a vertex
   * \tparam VertexProperty the type of data associated with the vertices
   * \tparam EdgeProperty the type of data associated with the edge
   *
   * \ingroup graph_types
   * \see Graph
   */
  template <typename Vertex,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class undirected_graph {

    // Private types
    //==========================================================================
  private:

    //! The map type used to associate neighbors and edge data with each vertex.
    typedef std::unordered_map<Vertex, EdgeProperty*> neighbor_map;

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
    typedef std::unordered_map<Vertex, vertex_data> vertex_data_map;

    // Public type declerations
    //==========================================================================
  public:
    typedef Vertex vertex_type;             //!< The vertex type.
    typedef undirected_edge<Vertex> edge_type; //!< The edge type.
    typedef VertexProperty vertex_property; //!< Data associated with vertices.
    typedef EdgeProperty edge_property;     //!< Data associated with edges.

    // Forward declerations. See the bottom of the class for implemenations
    class edge_iterator;     //!< Iterator over all edges of the graph.
    class in_edge_iterator;  //!< Iterator over incoming edges to a node.
    class out_edge_iterator; //!< Iterator over outgoing edges from a node.

    //! Iterator over all vertices.
    typedef map_key_iterator<vertex_data_map> vertex_iterator;

    //! Iterator over neighbors of a single vertex.
    typedef map_key_iterator<neighbor_map> neighbor_iterator;

    // Constructors and destructors
    //==========================================================================
  public:
    //! Create an empty graph.
    undirected_graph()
      : edge_count_(0) { }

    //! Create a graph from a list of pairs of vertices.
    template <typename Range>
    explicit undirected_graph(const Range& edges, typename Range::iterator* = 0)
      : edge_count_(0) {
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
      edge_count_ = g.edge_count_;
      for (edge_type e : edges()) {
        if(e.property_ != nullptr) {
          edge_property* ptr =
            new edge_property(*static_cast<edge_property*>(e.property_));
          data_[e.source()].neighbors[e.target()] = ptr;
          data_[e.target()].neighbors[e.source()] = ptr;
        }
      }
      return *this;
    }

    //! Swaps two graphs in constant time.
    friend void swap(undirected_graph& a, undirected_graph& b) {
      swap(a.data_, b.data_);
      std::swap(a.edge_count_, b.edge_count_);
    }

    // Accessors
    //==========================================================================

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return { vertex_iterator(data_.begin()),
               vertex_iterator(data_.end()) };
    }

    //! Returns the vertices adjacent to u.
    iterator_range<neighbor_iterator>
    neighbors(Vertex u) const {
      const vertex_data& data(find_vertex_data(u));
      return { neighbor_iterator(data.neighbors.begin()),
               neighbor_iterator(data.neighbors.end()) };
    }

    //! Returns the range of all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { edge_iterator(data_.begin(), data_.end()),
               edge_iterator(data_.end(), data_.end()) };
    }

    //! Returns all edges connected to u.
    iterator_range<out_edge_iterator>
    edges(Vertex u) const {
      const vertex_data& data(find_vertex_data(u));
      return { out_edge_iterator(u, data.neighbors.begin()),
               out_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(Vertex u) const {
      const vertex_data& data(find_vertex_data(u));
      return { in_edge_iterator(u, data.neighbors.begin()),
               in_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns the edges outgoing from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(Vertex u) const {
      const vertex_data& data(find_vertex_data(u));
      return { out_edge_iterator(u, data.neighbors.begin()),
               out_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(Vertex u) const {
      return data_.find(u) != data_.end();
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(Vertex u, Vertex v) const {
      auto it = data_.find(u);
      return it != data_.end() &&
        (it->second.neighbors.find(v) != it->second.neighbors.end());
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(const edge_type& e) const {
      return contains(e.source(), e.target());
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    edge_type edge(Vertex u,  Vertex v) const {
      const vertex_data& data = find_vertex_data(u);
      auto it = data.neighbors.find(v);
      assert(it != data.neighbors.end());
      return edge_type(u, v, it->second);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t in_degree(Vertex u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t out_degree(Vertex u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t degree(Vertex u) const {
      return find_vertex_data(u).neighbors.size();
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
      return edge_count_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    edge_type reverse(const edge_type& e) const {
      return e.reverse();
    }

    //! Returns the property associated with a vertex,
    const VertexProperty& operator[](Vertex u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](Vertex u) {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with an edge.
    const EdgeProperty& operator[](const edge_type& e) const {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](const edge_type& e) {
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

    // Modifications
    //==========================================================================
    /**
     * Adds a vertex to a graph and associate the property with that vertex.
     * If the vertex is already present, its property is not overwritten.
     * \returns true if the insertion took place (i.e., vertex was not present).
     */
    bool add_vertex(Vertex u, const VertexProperty& p = VertexProperty()) {
      assert(u != Vertex());
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
    std::pair<edge_type, bool>
    add_edge(Vertex u, Vertex v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != Vertex());
      assert(v != Vertex());
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
      ++edge_count_;
      return std::make_pair(edge_type(u, v, ptr), true);
    }

    //! Removes a vertex from the graph and all its incident edges.
    void remove_vertex(Vertex u) {
      remove_edges(u);
      data_.erase(u);
    }

    //! Removes an undirected edge {u, v}.
    void remove_edge(Vertex u, Vertex v) {
      vertex_data& data = find_vertex_data(u);
      auto it = data.neighbors.find(v);
      assert(it != data.neighbors.end());

      // delete the edge data and the edge itself
      delete it->second;
      data.neighbors.erase(it);
      data_[v].neighbors.erase(u);
      --edge_count_;
    }

    //! Removes all edges incident to a vertex.
    void remove_edges(Vertex u) {
      neighbor_map& neighbors = find_vertex_data(u).neighbors;
      edge_count_ -= neighbors.size();
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
      edge_count_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data_.clear();
      edge_count_ = 0;
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

    // Private functions
    //==========================================================================
  private:
    const vertex_data& find_vertex_data(Vertex v) const {
      auto it = data_.find(v);
      assert(it != data_.end());
      return it->second;
    }

    vertex_data& find_vertex_data(Vertex v) {
      auto it = data_.find(v);
      assert(it != data_.end());
      return it->second;
    }

    void free_edge_data() {
      for (edge_type e : edges()) {
        if(e.property_) {
          delete static_cast<EdgeProperty*>(e.property_);
        }
      }
    }

    // Implementation of edge iterators
    //==========================================================================
  public:
    class edge_iterator :
      public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename vertex_data_map::const_iterator primary_iterator;
      typedef typename neighbor_map::const_iterator secondary_iterator;

      edge_iterator() {}

      edge_iterator(primary_iterator it1, primary_iterator end1)
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

      primary_iterator it1_;   //!< the iterator to the vertex data
      primary_iterator end1_;  //!< the iterator past the last vertex data
      secondary_iterator it2_; //!< the iterator to the current neighbor

    }; // class edge_iterator

    class in_edge_iterator :
      public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename neighbor_map::const_iterator iterator;

      in_edge_iterator() { }

      in_edge_iterator(Vertex target, iterator it)
        : target_(target), it_(it) { }

      edge_type operator*() const {
        return edge_type(it_->first, target_, it_->second);
      }

      in_edge_iterator& operator++() {
        ++it_;
        return *this;
      }

      in_edge_iterator operator++(int) {
        in_edge_iterator copy(*this);
        ++it_;
        return copy;
      }

      bool operator==(const in_edge_iterator& o) const {
        return it_ == o.it_;
      }

      bool operator!=(const in_edge_iterator& o) const {
        return it_ != o.it_;
      }

    private:
      Vertex target_; //!< the target vertex
      iterator it_;   //!< the iterator to the current neighbor

    }; // class in_eddge_iterator

    class out_edge_iterator :
      public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename neighbor_map::const_iterator iterator;

      out_edge_iterator() {}

      out_edge_iterator(Vertex source, iterator it)
        : source_(source), it_(it) { }

      edge_type operator*() const {
        return edge_type(source_, it_->first, it_->second);
      }

      out_edge_iterator& operator++() {
        ++it_;
        return *this;
      }

      out_edge_iterator operator++(int) {
        out_edge_iterator copy(*this);
        ++it_;
        return copy;
      }

      bool operator==(const out_edge_iterator& o) const {
        return it_ == o.it_;
      }

      bool operator!=(const out_edge_iterator& o) const {
        return it_ != o.it_;
      }

    private:
      Vertex source_; //!< the source vertex
      iterator it_;   //!< the iterator to the current neighbor

    }; // class out_edge_iterator

    // Private data
    //==========================================================================
  private:
    //! A map from each vertex to its vertex data.
    vertex_data_map data_;

    //! The total number of edges in the graph.
    std::size_t edge_count_;

  }; // class undirected_graph

  /**
   * Prints the graph to an output stream
   * \relates undirected_graph
   */
  template <typename Vertex, typename VP, typename EP>
  std::ostream& operator<<(std::ostream& out,
                           const undirected_graph<Vertex, VP, EP>& g) {
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

    static vertex_descriptor null_vertex() { return vertex_descriptor(); }

  };

} // namespace boost

#endif
