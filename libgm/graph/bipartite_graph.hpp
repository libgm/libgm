#ifndef LIBGM_BIPARTITE_GRAPH_HPP
#define LIBGM_BIPARTITE_GRAPH_HPP

#include <libgm/global.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/serialize.hpp>

#include <iterator>
#include <map>
#include <random>
#include <unordered_map>

namespace libgm {

  /**
   * A class that represents an undirected bipartite graph. The graph contains
   * two types of vertices (type 1 and type 2), which correspond to the two
   * sides of the partition. These vertices are represented by the template
   * arguments Vertex1 and Vertex2, respectively. These two types _must_ be
   * distinct and not convertible to each other. This requirement is necessary
   * to allow for overload resolution to work in functions, such as neighbors().
   * The class defines the member type vertex, which is effectively a union
   * of the two vertex types.
   *
   * \tparam Vertex1 the type that represents a type-1 vertex
   * \tparam Vertex2 the type that represents a type-2 vertex
   * \tparam VertexProperty the type of data associated with the vertices
   * \tparam EdgeProperty the type of data associated with the edge
   */
  template <typename Vertex1,
            typename Vertex2,
            typename Vertex1Property = void_,
            typename Vertex2Property = void_,
            typename EdgeProperty = void_>
  class bipartite_graph {

    // Private types
    //==========================================================================
  private:
    typedef std::unordered_map<Vertex1, EdgeProperty*> neighbor1_map;
    typedef std::unordered_map<Vertex2, EdgeProperty*> neighbor2_map;

    struct vertex1_data {
      Vertex1Property property;
      neighbor2_map neighbors;
      vertex1_data() : property() { }
    };

    struct vertex2_data {
      Vertex2Property property;
      neighbor1_map neighbors;
      vertex2_data() : property() { }
    };

    typedef std::unordered_map<Vertex1, vertex1_data> vertex1_data_map;
    typedef std::unordered_map<Vertex2, vertex2_data> vertex2_data_map;

    // Public types
    //==========================================================================
  public:
    // vertex and edge types
    typedef Vertex1 vertex1_type;
    typedef Vertex2 vertex2_type;
    class edge_type;

    // properties
    typedef Vertex1Property vertex1_property;
    typedef Vertex2Property vertex2_property;
    typedef EdgeProperty edge_property;
    
    // vertex iterators
    typedef map_key_iterator<vertex1_data_map> vertex1_iterator;
    typedef map_key_iterator<vertex2_data_map> vertex2_iterator;
    typedef map_key_iterator<neighbor1_map> neighbor1_iterator;
    typedef map_key_iterator<neighbor2_map> neighbor2_iterator;
    
    // edge iterators (forward declarations and typedefs)
    class edge_iterator;
    template <typename Target, typename Neighbors> class in_edge_iterator;
    template <typename Source, typename Neighbors> class out_edge_iterator;
    typedef in_edge_iterator<Vertex1, neighbor2_map> in1_edge_iterator;
    typedef in_edge_iterator<Vertex2, neighbor1_map> in2_edge_iterator;
    typedef out_edge_iterator<Vertex1, neighbor2_map> out1_edge_iterator;
    typedef out_edge_iterator<Vertex2, neighbor1_map> out2_edge_iterator;
    
    // Constructors, destructors, and related functions
    //==========================================================================
  public:
    //! Creates an empty graph.
    bipartite_graph()
      : edge_count_(0) { }

    //! Creates a graph from a range of vertex pairs
    template <typename Range>
    explicit bipartite_graph(const Range& edges, typename Range::iterator* = 0)
      : edge_count_(0) {
      for (const std::pair<Vertex1, Vertex2>& p : edges) {
        add_edge(p.first, p.second);
      }
    }

    //! Copy constructor
    bipartite_graph(const bipartite_graph& g) {
      *this = g;
    }

    //! Destructor
    ~bipartite_graph() {
      free_edge_data();
    }

    //! Assignment
    bipartite_graph& operator=(const bipartite_graph& g) {
      if (this == &g) { return *this; }
      free_edge_data();
      data1_ = g.data1_;
      data2_ = g.data2_;
      edge_count_ = g.edge_count_;
      for (edge_type e : edges()) {
        edge_property* ptr = new edge_property(*e.property_);
        data1_[e.v1()].neighbors[e.v2()] = ptr;
        data2_[e.v2()].neighbors[e.v1()] = ptr;
      }
      return *this;
    }

    //! Swap with another graph in constant time
    friend void swap(bipartite_graph& a, bipartite_graph& b) {
      swap(a.data1_, b.data1_);
      swap(a.data2_, b.data2_);
      std::swap(a.edge_count_, b.edge_count_);
    }

    // Accessors
    //==========================================================================
  public:
    //! Returns the range of all type-1 vertices
    iterator_range<vertex1_iterator>
    vertices1() const {
      return { vertex1_iterator(data1_.begin()),
               vertex1_iterator(data1_.end()) };
    }

    //! Returns the range of all type-2 vertices
    iterator_range<vertex2_iterator>
    vertices2() const {
      return { vertex2_iterator(data2_.begin()),
               vertex2_iterator(data2_.end()) };
    }

    //! Returns the type-2 vertices adjacent to a type-1 vertex
    iterator_range<neighbor2_iterator>
    neighbors(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return { neighbor2_iterator(data.neighbors.begin()),
               neighbor2_iterator(data.neighbors.end()) };
    }

    //! Returns the type-1 vertices adjacent to a type-2 vertex
    iterator_range<neighbor1_iterator>
    neighbors(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return { neighbor1_iterator(data.neighbors.begin()),
               neighbor1_iterator(data.neighbors.end()) };
    }

    //! Returns all edges in the graph
    iterator_range<edge_iterator>
    edges() const {
      return { edge_iterator(data1_.begin(), data1_.end()),
               edge_iterator(data1_.end(), data1_.end()) };
    }

    //! Returns the edges incoming to a type-1 vertex
    iterator_range<in1_edge_iterator>
    in_edges(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return { in1_edge_iterator(u, data.neighbors.begin()),
               in1_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns the edges incoming to a type-2 vertex
    iterator_range<in2_edge_iterator>
    in_edges(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return { in2_edge_iterator(u, data.neighbors.begin()),
               in2_edge_iterator(u, data.neighbors.end()) };
    }
    
    //! Returns the edges outgoing from a type-1 vertex
    iterator_range<out1_edge_iterator>
    out_edges(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return { out1_edge_iterator(u, data.neighbors.begin()),
               out1_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns the edges outgoing from a type-2 vertex
    iterator_range<out2_edge_iterator>
    out_edges(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return { out2_edge_iterator(u, data.neighbors.begin()),
               out2_edge_iterator(u, data.neighbors.end()) };
    }

    //! Returns true if the graph contains the given type-1 vertex
    bool contains(Vertex1 u) const {
      return data1_.find(u) != data1_.end();
    }

    //! Returns true if the graph contains the given type-2 vertex
    bool contains(Vertex2 u) const {
      return data2_.find(u) != data2_.end();
    }

    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(Vertex1 u, Vertex2 v) const {
      auto it = data1_.find(u);
      return it != data1_.end() &&
        it->second.neighbors.find(v) != it->second.neighbors.end();
    }

    //! Returns true if the graph contains an undirected edge
    bool contains(const edge_type& e) const {
      return contains(e.v1(), e.v2());
    }
    
    //! Returns an undirected edge between u and v. The edge must exist.
    edge_type edge(Vertex1 u, Vertex2 v) const {
      const vertex1_data& data = find_vertex_data(u);
      typename neighbor2_map::const_iterator it = data.neighbors.find(v);
      assert(it != data.neighbors.end());
      return edge_type(u, v, it->second);
    }

    //! Returns an undirected edge between u and v. The edge must exist.
    edge_type edge(Vertex2 u, Vertex1 v) const {
      const vertex2_data& data = find_vertex_data(u);
      typename neighbor1_map::const_iterator it = data.neighbors.find(v);
      assert(it != data.neighbors.end());
      return edge_type(u, v, it->second);
    }

    //! Returns the number of edges connected to a type-1 vertex
    size_t degree(Vertex1 u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges connected to a type-2 vertex
    size_t degree(Vertex2 u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return data1_.empty() && data2_.empty();
    }

    //! Returns the number of type-1 vertices
    size_t num_vertices1() const {
      return data1_.size();
    }

    //! Returns the number of type-2 vertices
    size_t num_vertices2() const {
      return data2_.size();
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return data1_.size() + data2_.size();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return edge_count_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
    edge_type reverse(const edge_type& e) const {
      return e.reverse();
    }

    //! Returns the property associated with a type-1 vertex
    const Vertex1Property& operator[](Vertex1 u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a type-2 vertex
    const Vertex2Property& operator[](Vertex2 u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a type-1 vertex
    Vertex1Property& operator[](Vertex1 u) {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a type-2 vertex
    Vertex2Property& operator[](Vertex2 u) {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with an edge
    const EdgeProperty& operator[](const edge_type& e) const {
      return *e.property_;
    }

    //! Returns the property associated with an edge
    EdgeProperty& operator[](const edge_type& e) {
      return *e.property_;
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge must exist.
     */
    const EdgeProperty& operator()(Vertex1 u, Vertex2 v) const {
      return *edge(u, v).property_;
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge is added if necessary.
     */
    EdgeProperty& operator()(Vertex1 u, Vertex2 v) {
      return *add_edge(u, v).first.property_;
    }

    /**
     * Draws a random type-1 vertex in the graph, assuming one exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    Vertex1 sample_vertex1(Generator& rng) const {
      return sample_vertex(data1_, rng);
    }
    
    /**
     * Draws a random type-2 vertex in the graph, assumign on exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    Vertex2 sample_vertex2(Generator& rng) const {
      return sample_vertex(data2_, rng);
    }

    /**
     * Compares the graph structure and the vertex & edge properties.
     * The property types must support operator!=().
     */
    bool operator==(const bipartite_graph& g) const {
      if (num_vertices1() != g.num_vertices1() ||
          num_vertices2() != g.num_vertices2() || 
          num_edges() != g.num_edges()) {
        return false;
      }
      for (const auto& vp : data1_) {
        auto vit = g.data1_.find(vp.first);
        if (vit == g.data1_.end() ||
            vp.second.property != vit->second.property) {
          return false;
        }
        const neighbor2_map& neighbors = vit->second.neighbors;
        for (const auto& ep : vp.second.neighbors) {
          auto eit = neighbors.find(ep.first);
          if (eit == neighbors.end() || *ep.second != *eit->second) {
            return false;
          }
        }
      }
      for (const auto& vp : data2_) {
        auto vit = g.data2_.find(vp.first);
        if (vit == g.data2_.end() ||
            vp.second.property != vit->second.property) {
          return false;
        }
      }
      return true;
    }

    //! Compares the graph structure and the vertex & edge properties
    bool operator!=(const bipartite_graph& g) const {
      return !operator==(g);
    }

    //! Prints the graph to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const bipartite_graph& g) {
      out << "Type-1 vertices" << std::endl;
      for (Vertex1 u : g.vertices1()) {
        out << u << ": " << g[u] << std::endl;
      }
      out << "Type-2 vertices" << std::endl;
      for (Vertex2 u : g.vertices2()) {
        out << u << ": " << g[u] << std::endl;
      }
      out << "Edges" << std::endl;
      for (edge_type e : g.edges()) {
        out << e << std::endl;
      }
      return out;
    }

    //! Prints the degree distribution for the given vertex range
    template <typename Range>
    void print_degree_distribution(std::ostream& out, const Range& range) const {
      std::map<size_t, size_t> count;
      for (auto v : range) {
        ++count[degree(v)];
      }
      for (const auto& p : count) {
        std::cout << p.first << ' ' << p.second << std::endl;
      }
    }

    // Modifications
    //==========================================================================
    /**
     * Adds a type-1 vertex to a graph and associates a property with
     * that vertex. If the vertex is already present, its property is
     * not overwritten.
     * \return true if the insertion took place (i.e., vertex was not present).
     */
    bool add_vertex(Vertex1 u, const Vertex1Property& p = Vertex1Property()) {
      assert(u != Vertex1());
      if (contains(u)) {
        return false;
      } else {
        data1_[u].property = p;
        return true;
      }
    }

    /**
     * Adds a type-2 vertex to a graph and associates a property with that
     * that vertex. If the vertex is already present, its property is
     * not overwritten.
     * \return true if the insertion took place (i.e., vertex was not present).
     */
    bool add_vertex(Vertex2 u, const Vertex2Property& p = Vertex2Property()) {
      assert(u != Vertex2());
      if (contains(u)) {
        return false;
      } else {
        data2_[u].property = p;
        return true;
      }
    }

    /**
     * Adds an edge {u, v} to the graph. If the edge already exists, its
     * property is not overwritten. If u or v are not present, they are added.
     * \return the edge and bool indicating whether the edge was newly added.
     */
    std::pair<edge_type, bool>
    add_edge(Vertex1 u, Vertex2 v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != Vertex1());
      assert(v != Vertex2());
      auto uit = data1_.find(u);
      if (uit != data1_.end()) {
        auto vit = uit->second.neighbors.find(v);
        if (vit != uit->second.neighbors.end()) {
          return std::make_pair(edge_type(u, v, vit->second), false);
        }
      }
      EdgeProperty* ptr = new EdgeProperty(p);
      data1_[u].neighbors[v] = ptr;
      data2_[v].neighbors[u] = ptr;
      ++edge_count_;
      return std::make_pair(edge_type(u, v, ptr), true);
    }

    //! Removes a type-1 vertex and all its incident edges from the graph.
    void remove_vertex(Vertex1 u) {
      remove_edges(u);
      data1_.erase(u);
    }

    //! Removes a type-2 vertex and all its incident edges from the graph.
    void remove_vertex(Vertex2 u) {
      remove_edges(u);
      data2_.erase(u);
    }

    //! Removes an undirected edge {u, v}. The edge must be present.
    void remove_edge(Vertex1 u, Vertex2 v) {
      // find the edge (u, v)
      vertex1_data& data = find_vertex_data(u);
      auto it = data.neighbors.find(v);
      assert(it != data.neighbors.end());

      // delete the edge data and the two symmetric edges
      delete it->second;
      data.neighbors.erase(it);
      data2_[v].neighbors.erase(u);
      --edge_count_;
    }

    //! Removes all edges incident to a type-1 vertex.
    void remove_edges(Vertex1 u) {
      neighbor2_map& neighbors = find_vertex_data(u).neighbors;
      edge_count_ -= neighbors.size();
      for (const auto& p : neighbors) {
        data2_[p.first].neighbors.erase(u);
        delete p.second;
      }
      neighbors.clear();
    }

    //! Removes all edges incident to a type-2 vertex.
    void remove_edges(Vertex2 u) {
      neighbor1_map& neighbors = find_vertex_data(u).neighbors;
      edge_count_ -= neighbors.size();
      for (const auto& p : neighbors) {
        data1_[p.first].neighbors.erase(u);
        delete p.second;
      }
      neighbors.clear();
    }

    //! Removes all edges from the graph.
    void remove_edges() {
      free_edge_data();
      for (typename vertex1_data_map::reference p : data1_) {
        p.second.neighbors.clear();
      }
      for (typename vertex2_data_map::reference p : data2_) {
        p.second.neighbors.clear();
      }
      edge_count_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data1_.clear();
      data2_.clear();
      edge_count_ = 0;
    }

    //! Saves the graph to an archive.
    void save(oarchive& ar) const {
      ar << num_vertices1() << num_vertices2() << num_edges();
      for (typename vertex1_data_map::const_reference p : data1_) {
        ar << p.first << p.second.property;
      }
      for (typename vertex2_data_map::const_reference p : data2_) {
        ar << p.first << p.second.property;
      }
      for (edge_type e : edges()) {
        ar << e.v1() << e.v2() << *e.property_;
      }
    }

    //! Loads the graph from an archive.
    void load(iarchive& ar) {
      clear();
      size_t num_vertices1, num_vertices2, num_edges;
      Vertex1 u;
      Vertex2 v;
      ar >> num_vertices1 >> num_vertices2 >> num_edges;
      while (num_vertices1-- > 0) {
        ar >> u;
        ar >> data1_[u].property;
      }
      while (num_vertices2-- > 0) {
        ar >> v;
        ar >> data2_[v].property;
      }
      while (num_edges-- > 0) {
        ar >> u >> v;
        ar >> operator()(u, v);
      }
    }

    // Private helper functions
    //==========================================================================
  private: 
    const vertex1_data& find_vertex_data(Vertex1 u) const { 
      typename vertex1_data_map::const_iterator it = data1_.find(u);
      assert(it != data1_.end());
      return it->second;
    }

    const vertex2_data& find_vertex_data(Vertex2 u) const { 
      typename vertex2_data_map::const_iterator it = data2_.find(u);
      assert(it != data2_.end());
      return it->second;
    }

    vertex1_data& find_vertex_data(Vertex1 u) { 
      typename vertex1_data_map::iterator it = data1_.find(u);
      assert(it != data1_.end());
      return it->second;
    }

    vertex2_data& find_vertex_data(Vertex2 u) { 
      typename vertex2_data_map::iterator it = data2_.find(u);
      assert(it != data2_.end());
      return it->second;
    }

    template <typename DataMap, typename Generator>
    static typename DataMap::key_type
    sample_vertex(DataMap& data, Generator& rng) {
      assert(!data.empty());
      std::uniform_int_distribution<size_t> unifb(0, data.bucket_count() - 1);
      while (true) {
        size_t bucket = unifb(rng);
        size_t bsize = data.bucket_size(bucket);
        if (bsize > 0) {
          std::uniform_int_distribution<size_t> unifi(0, bsize - 1);
          return std::next(data.begin(bucket), unifi(rng))->first;
        }
      }
    }

    void free_edge_data() {
      for (edge_type e : edges()) {
        if(e.property_) {
          delete e.property_;
        }
      }
    }

    // Implementation of edge type and edge iterators
    //==========================================================================
  public:
    class edge_type {
      Vertex1 v1_;
      Vertex2 v2_;
      bool forward_; // true if the edge is from type1 to type2
      EdgeProperty* property_;
      friend class bipartite_graph;
    public:
      edge_type()
        : v1_(), v2_(), forward_(true), property_() { }
      edge_type(Vertex1 v1, Vertex2 v2, EdgeProperty* property)
        : v1_(v1), v2_(v2), forward_(true), property_(property) { }
      edge_type(Vertex2 v2, Vertex1 v1, EdgeProperty* property)
        : v1_(v1), v2_(v2), forward_(false), property_(property) { }
      explicit operator bool() const {
        return v1_ != Vertex1() || v2_ != Vertex2();
      }
      Vertex1 v1() const {
        return v1_;
      }
      Vertex2 v2() const {
        return v2_;
      }
      std::pair<Vertex1, Vertex2> endpoints() const {
        return std::make_pair(v1_, v2_);
      }      
      bool forward() const {
        return forward_;
      }
      bool operator==(const edge_type& o) const {
        return v1_ == o.v1_ && v2_ == o.v2_;
      }
      bool operator!=(const edge_type& o) const {
        return v1_ != o.v1_ || v2_ != o.v2_;
      }
      edge_type reverse() const {
        edge_type e = *this;
        e.forward_ = !e.forward_;
        return e;
      }
      friend std::ostream& operator<<(std::ostream& out, const edge_type& e) {
        if (e.forward()) {
          out << e.v1() << " -- " << e.v2();
        } else {
          out << e.v2() << " -- " << e.v1();
        }
        return out;
      }
      friend size_t hash_value(const edge_type& e) {
        return boost::hash_value(std::make_pair(e.v1_, e.v2_));
      }
    }; // class edge_type

    class edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference;
      typedef typename vertex1_data_map::const_iterator primary_iterator;
      typedef typename neighbor2_map::const_iterator secondary_iterator;

      edge_iterator() { }

      edge_iterator(primary_iterator it1, primary_iterator end1)
        : it1_(it1), end1_(end1) {
        // skip all the empty neighbor maps
        while (it1_ != end1_ && it1_->second.neighbors.empty()) {
          ++it1_;
        }
        // if not reached the end, initialize the secondary iterator
        if (it1_ != end1_) {
          it2_ = it1_->second.neighbors.begin();
        }
      }

      edge_type operator*() const {
        return edge_type(it1_->first, it2_->first, it2_->second);
      }

      edge_iterator& operator++() {
        ++it2_;
        if (it2_ == it1_->second.neighbors.end()) {
          // at the end of the neighbor map; advance the primary iterator
          do {
            ++it1_;
          } while (it1_ != end1_ && it1_->second.neighbors.empty());
          if (it1_ != end1_) {
            it2_ = it1_->second.neighbors.begin();
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
        return !(*this == other);
      }

    private:
      primary_iterator it1_;   //!< the iterator to the current vertex data
      primary_iterator end1_;  //!< the iterator past the last vertex data
      secondary_iterator it2_; //!< the iterator to the current neighbor

    }; // class edge_iterator

    template <typename Target, typename Neighbors>
    class in_edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {
    public:
      typedef edge_type reference; // override the base class reference type
      typedef typename Neighbors::const_iterator iterator; // base iterator type

      in_edge_iterator() { }

      in_edge_iterator(Target target, iterator it)
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
      Target target_; //!< the target vertex
      iterator it_;   //!< the iterator to the current neighbor

    }; // class in_edge_iterator

    template <typename Source, typename Neighbors>
    class out_edge_iterator
      : public std::iterator<std::forward_iterator_tag, edge_type> {

    public:
      typedef edge_type reference; // override the base class reference type
      typedef typename Neighbors::const_iterator iterator; // base iterator type

    public:
      out_edge_iterator() { }

      out_edge_iterator(Source source, iterator it)
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
      Source source_; //!< the source vertex
      iterator it_;   //!< the iterator to the current neighbor

    }; // class out_edge_iterator

    // Private data
    //==========================================================================
  private:
    //! The properties and neighbors of type-1 vertices.
    vertex1_data_map data1_;

    //! The properties and neighbors of type-2 vertices.
    vertex2_data_map data2_;

    //! The number of edges.
    size_t edge_count_;
      
  }; // class bipartite_graph

} // namespace libgm

#endif
