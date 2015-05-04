#ifndef LIBGM_DIRECTED_MULTIGRAPH_HPP
#define LIBGM_DIRECTED_MULTIGRAPH_HPP

#include <libgm/global.hpp>
#include <libgm/datastructure/vector_map.hpp>
#include <libgm/graph/directed_edge.hpp> 
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/serialize.hpp>

#include <boost/graph/graph_traits.hpp>

#include <algorithm>
#include <iterator>
#include <iosfwd>
#include <unordered_map>

namespace libgm {

  /**
   * A directed graph that permits multiple edges between a pair of vertices.
   * The edge data is stored in an std::vector (using a vector_map).
   *
   * Inserting an edge invalidates the edge and neighbor iterators.
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
  class directed_multigraph {

    // Private types
    //==========================================================================
  private:
    //! The map type used to associate neighbors and edge data with each vertex.
    typedef vector_map<Vertex, EdgeProperty*> neighbor_map;

    /**
     * A struct with the data associated with each vertex. This structure
     * stores the property associated with the vertex as well all edges from/to
     * parent and child vertices (along with the edge properties).
     */
    struct vertex_data {
      VertexProperty property;
      neighbor_map parents;
      neighbor_map children;
      vertex_data() : property() { }
    };

    //! The map types that associates all the vertices with their vertex_data.
    typedef std::unordered_map<Vertex, vertex_data> vertex_data_map;

    // Public type declerations
    //==========================================================================
  public:
    typedef Vertex vertex_type;              //!< The vertex type.
    typedef directed_edge<Vertex> edge_type; //!< The edge type.
    typedef VertexProperty vertex_property;  //!< Data associated with vertices.
    typedef EdgeProperty edge_property;      //!< Data associated with Edges.
    
    // Forward declerations. See the bottom of the class for implemenations
    class edge_iterator;     //!< Iterator over all edges of the graph.
    class in_edge_iterator;  //!< Iterator over incoming edges to a node.
    class out_edge_iterator; //!< Iterator over outgoing edges from a node.
    
    //! Iterator over all vertices.
    typedef map_key_iterator<vertex_data_map> vertex_iterator;

    //! Iterator over the neighbors of a single vertex.
    typedef map_key_iterator<neighbor_map> neighbor_iterator;

    // Constructors and destructors
    //==========================================================================
  public:
    //! Create an empty graph.
    directed_multigraph()
      : edge_count_(0) { }

    //! Create a graph from a list of pairs.  
    template <typename Range>
    explicit directed_multigraph(const Range& edges,
                                 typename Range::iterator* = 0)
      : edge_count_(0) {
      for (std::pair<Vertex, Vertex> vp : edges) {
        add_edge(vp.first, vp.second);
      }
      sort_edges();
    }
    
    //! Copy constructor
    directed_multigraph(const directed_multigraph& g) {
      *this = g;
    }

    //! Destructor
    ~directed_multigraph() {
      free_edge_data();
    }

    //! Assignment operator
    directed_multigraph& operator=(const directed_multigraph& other) {
      // Destroy the information associated with this graph
      clear();
      for (vertex_type v : other.vertices()) {
        add_vertex(v, other[v]);
      }
      for (edge_type e : other.edges()) {
        add_edge(e.source(), e.target(), other[e]);
      }
      return *this;
    }

    //! Swaps two graphs in constant time.
    friend void swap(directed_multigraph& a, directed_multigraph& b) {
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
    
    //! Returns the parents of u.
    iterator_range<neighbor_iterator>
    parents(Vertex u) const {
      const vertex_data& data = find_vertex_data(u);
      return { neighbor_iterator(data.parents.begin()),
               neighbor_iterator(data.parents.end()) };
    }

    //! Returns the children of u.
    iterator_range<neighbor_iterator>
    children(Vertex u) const {
      const vertex_data& data = find_vertex_data(u);
      return { neighbor_iterator(data.children.begin()),
               neighbor_iterator(data.children.end()) };
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { edge_iterator(data_.begin(), data_.end()),
               edge_iterator(data_.end(), data_.end()) };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(Vertex u) const {
      const vertex_data& data = find_vertex_data(u);
      return { in_edge_iterator(u, data.parents.begin()),
               in_edge_iterator(u, data.parents.end()) };
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(Vertex u) const {
      const vertex_data& data = find_vertex_data(u);
      return { out_edge_iterator(u, data.children.begin()),
               out_edge_iterator(u, data.children.end()) };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(Vertex u) const {
      return data_.find(u) != data_.end();
    }
    
    //! Returns true if the graph contains a directed edge (u, v).
    bool contains(Vertex u, Vertex v) const {
      auto it = data_.find(u);
      return it != data_.end() && 
        (it->second.children.find(v) != it->second.children.end());     
    }
    
    //! Returns true if the graph contains a directed edge.
    bool contains(const edge_type& e) const {
      return contains(e.source(), e.target());
    }

    //! Returns a directed edge (u,v) between two vertices. The edge must exist.
    edge_type edge(Vertex u, Vertex v) const {
      const vertex_data& data = find_vertex_data(u);
      auto it = data.children.find(v);
      assert(it != data.children.end());
      return edge_type(u, v, it->second);
    }

    /**
     * Returns the number of incoming edges to a vertex. The vertex must
     * already be present in the graph.
     */
    size_t in_degree(Vertex u) const {
      return find_vertex_data(u).parents.size();
    }

    /**
     * Returns the number of outgoing edges to a vertex. The vertex must
     * already be present in the graph.
     */
    size_t out_degree(Vertex u) const {
      return find_vertex_data(u).children.size();
    }
    
    /**
     * Returns the total number of edges adjacent to a vertex. The vertex must
     * already be present in the graph.
     */
    size_t degree(Vertex u) const {
      const vertex_data& data = find_vertex_data(u);
      return data.parents.size() + data.children.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data_.empty();
    }

    //! Returns the number of vertices.
    size_t num_vertices() const {
      return data_.size();
    }

    //! Returns the number of edges.
    size_t num_edges() const {
      return edge_count_;
    }

    //! Given a directed edge (u, v), returns a directed edge (v, u).
    //! The edge (v, u) must exist.
    edge_type reverse(const edge_type& e) const { 
      return edge(e.target(), e.source()); 
    }

    //! Returns the property associated with a vertex.
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
     * Returns the property associated with edge (u, v).
     * The edge must exist.
     */
    const EdgeProperty& operator()(Vertex u, Vertex v) const {
      return *static_cast<EdgeProperty*>(edge(u, v).property_);
    }

    /**
     * Compares the graph structure and the vertex & edge properties.
     * The property type must support operator!= and all the properties
     * for the same (multiple) edge must be equal.
     */
    bool operator==(const directed_multigraph& other) const {
      if (num_vertices() != other.num_vertices() ||
          num_edges() != other.num_edges()) {
        return false;
      }
      auto edge_compare = [](const std::pair<Vertex, EdgeProperty*>& a,
                             const std::pair<Vertex, EdgeProperty*>& b) {
        return a.first == b.first && *a.second == *b.second;
      };
      for (const auto& vp : data_) {
        auto vit = other.data_.find(vp.first);
        if (vit == other.data_.end() ||
            vp.second.property != vit->second.property) {
          return false;
        }
        const neighbor_map& children = vit->second.children;
        if (vp.second.children.size() != children.size() ||
            !std::equal(children.begin(), children.end(),
                        vp.second.children.begin(), edge_compare)) {
          return false;
        }
      }
      return true;
    }

    //! Inequality comparison
    bool operator!=(const directed_multigraph& other) const {
      return !(*this == other);
    }

    // Modifications
    //==========================================================================
    /**
     * Adds a vertex to the graph and associates a property with that vertex.
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
     * Adds an edge (u, v) to the graph.
     * If u and v are not present, they are added.
     * \return the edge and true (the edge is always added).
     */
    std::pair<edge_type, bool>
    add_edge(Vertex u, Vertex v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != Vertex());
      assert(v != Vertex());
      EdgeProperty* ptr = new EdgeProperty(p);
      data_[u].children.emplace(v, ptr);
      data_[v].parents.emplace(u, ptr);
      ++edge_count_;
      return std::make_pair(edge_type(u, v, ptr), true);
    }

    /**
     * Sorts the edges in the graph after they have been added.
     * This is needed for lookups and comparisons.
     */
    void sort_edges() {
      for (auto& p : data_) {
        p.second.children.sort();
        p.second.parents.sort();
      }
    }

    //! Removes a vertex from the graph and all its incident edges.
    void remove_vertex(Vertex u) {
      remove_edges(u);      
      data_.erase(u);
    }

    //! Removes all directed edges (u, v).
    void remove_edge(Vertex u, Vertex v) {
      vertex_data& data = find_vertex_data(u);
      typename neighbor_map::iterator begin, end;
      std::tie(begin, end) = data.children.equal_range(v);
      edge_count_ -= std::distance(begin, end);

      // delete the edge data and the edge itself
      for (auto it = begin; it != end; ++it) {
        delete it->second;
      }
      data.children.erase(begin, end);
      data_[v].parents.erase(u);
    }

    //! Removes all edges incident to a vertex.
    void remove_edges(Vertex u) {
      remove_in_edges(u);
      remove_out_edges(u);
    }

    //! Removes all edges incoming to a vertex.
    void remove_in_edges(Vertex u) {
      neighbor_map& parents = find_vertex_data(u).parents;
      edge_count_ -= parents.size();
      for (const auto& p : parents) {
        data_[p.first].children.erase(u);
        if (p.second) { delete p.second; }
      }
      parents.clear();
    }

    //! Removes all edges outgoing from a vertex.
    void remove_out_edges(Vertex u) {
      neighbor_map& children = find_vertex_data(u).children;
      edge_count_ -= children.size();
      for (const auto& c : children) {
        data_[c.first].parents.erase(u);
        if (c.second) { delete c.second; }
      }
      children.clear();
    }

    //! Removes all edges from the graph.
    void remove_edges() {
      free_edge_data();
      for (auto& p : data_) {
        p.second.parents.clear();
        p.second.children.clear();
      }
      edge_count_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data_.clear();
      edge_count_ = 0;
    }
    
    //! Saves the graph to an archive.
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
      size_t num_vertices, num_edges;
      Vertex u, v;
      ar >> num_vertices;
      ar >> num_edges;
      while (num_vertices-- > 0) {
        ar >> v;
        ar >> data_[v].property;
      }
      while (num_edges-- > 0) {
        ar >> u >> v; 
        ar >> operator[](add_edge(u, v).first);
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
        if (e.property_) {
          delete static_cast<EdgeProperty*>(e.property_);
        }
      }
    }

    // Implementation of edge iterators
    //==========================================================================
  public:
    class edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename vertex_data_map::const_iterator primary_iterator;
      typedef typename neighbor_map::const_iterator secondary_iterator;

      edge_iterator() { }

      edge_iterator(primary_iterator it1, primary_iterator end1)
        : it1_(it1), end1_(end1) {
        // skip all the empty children maps
        while (it1_ != end1_ && it1_->second.children.empty()) {
          ++it1_;
        }
        // if not reached the end, initialize the secondary iterator
        if (it1_ != end1_) {
          it2_ = it1_->second.children.begin();
        }
      }

      edge_type operator*() const {
        return edge_type(it1_->first, it2_->first, it2_->second);
      }

      edge_iterator& operator++() {
        ++it2_;
        if (it2_ == it1_->second.children.end()) {
          // at the end of the children map; advance the primary iterator
          do {
            ++it1_;
          } while (it1_ != end1_ && it1_->second.children.empty());
          if (it1_ != end1_) {
            it2_ = it1_->second.children.begin();
          }
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
      primary_iterator it1_;   //!< the iterator to the source vertex data
      primary_iterator end1_;  //!< the iterator past the last source data
      secondary_iterator it2_; //!< the iterator to the current neighbor

    }; // class edge_iterator

    class in_edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename neighbor_map::const_iterator iterator;

      in_edge_iterator(Vertex target, const iterator& it)
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
    }; // class in_edge_iterator

    class out_edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename neighbor_map::const_iterator iterator;

      out_edge_iterator(Vertex source, const iterator& it)
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
      iterator it_;   //!< The iterator to the current neighbor
    }; // class out_edge_iterator

    // Private data
    //==========================================================================
  private:
    //! A map from each vertex to its vertex data.
    vertex_data_map data_;
    
    //! The total number of directed edges in the graph.
    size_t edge_count_;

  }; // class directed_multigraph

  /**
   * Print graphs to an output stream.
   * \relates directed_multigraph
   */
  template <typename Vertex, typename VP, typename EP>
  std::ostream& operator<<(std::ostream& out,
                           const directed_multigraph<Vertex, VP, EP>& g) {
    out << "Vertices" << std::endl;
    for (Vertex v : g.vertices()) {
      out << v << ": " << g[v] << std::endl;
    }
    out << "Edges" << std::endl;
    for (directed_edge<Vertex> e : g.edges()) {
      out << e << std::endl;
    }
    return out;
  } 

} // namespace libgm

namespace boost {

  //! Type declarations that let our graph structure work in BGL algorithms
  template <typename Vertex, typename VP, typename EP>
  struct graph_traits< libgm::directed_multigraph<Vertex, VP, EP> > {
    
    typedef libgm::directed_multigraph<Vertex, VP, EP> graph_type;

    typedef typename graph_type::vertex             vertex_descriptor;
    typedef typename graph_type::edge               edge_descriptor;
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  adjacency_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;

    typedef directed_tag                            directed_category;
    typedef allow_parallel_edge_tag                 edge_parallel_category;

    struct traversal_category :
      public virtual boost::vertex_list_graph_tag,
      public virtual boost::incidence_graph_tag,
      public virtual boost::adjacency_graph_tag,
      public virtual boost::edge_list_graph_tag { };

    typedef size_t vertices_size_type;
    typedef size_t edges_size_type;
    typedef size_t degree_size_type;

    static vertex_descriptor null_vertex() { return vertex_descriptor(); }
  };

} // namespace boost

#endif
