#ifndef LIBGM_BIPARTITE_GRAPH_HPP
#define LIBGM_BIPARTITE_GRAPH_HPP

#include <libgm/graph/bipartite_edge.hpp>
#include <libgm/graph/util/sampling.hpp>
#include <libgm/graph/util/void.hpp>
#include <libgm/iterator/map_bind1_iterator.hpp>
#include <libgm/iterator/map_bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/iterator/map_map_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iterator>
#include <map>
#include <unordered_map>

namespace libgm {

  /**
   * A class that represents an undirected bipartite graph. The graph contains
   * two types of vertices (type 1 and type 2), which correspond to the two
   * sides of the partition. These vertices are represented by the template
   * arguments Vertex1 and Vertex2, respectively. These two types _must_ be
   * distinct and not convertible to each other. This requirement is necessary
   * to allow for overload resolution to work in functions, such as neighbors().
   *
   * \tparam Vertex1
   *         The type that represents a type-1 vertex.
   * \tparam Vertex2
   *         The type that represents a type-2 vertex.
   * \tparam Vertex1Property
   *         The type of data associated with type-1 vertices.
   * \tparam Vertex2Property
   *         The type of data associated with type-2 vertices.
   * \tparam EdgeProperty
   *         The type of data associated with edges.
   *
   * \ingroup graph_types
   */
  template <typename Vertex1,
            typename Vertex2,
            typename Vertex1Property = void_,
            typename Vertex2Property = void_,
            typename EdgeProperty = void_>
  class bipartite_graph {

    // Private types
    //--------------------------------------------------------------------------
  private:
    using neighbor1_map = std::unordered_map<Vertex2, EdgeProperty*>;
    using neighbor2_map = std::unordered_map<Vertex1, EdgeProperty*>;

    struct vertex1_data {
      Vertex1Property property;
      neighbor1_map neighbors;
      vertex1_data() : property() { }
    };

    struct vertex2_data {
      Vertex2Property property;
      neighbor2_map neighbors;
      vertex2_data() : property() { }
    };

    using vertex1_data_map = std::unordered_map<Vertex1, vertex1_data>;
    using vertex2_data_map = std::unordered_map<Vertex2, vertex2_data>;

    // Public types
    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge type, and properties
    using vertex1_type     = Vertex1;
    using vertex2_type     = Vertex2;
    using edge_type        = bipartite_edge<Vertex1, Vertex2>;
    using vertex1_property = Vertex1Property;
    using vertex2_property = Vertex2Property;
    using edge_property    = EdgeProperty;

    // Iterators
    using vertex1_iterator   = map_key_iterator<vertex1_data_map>;
    using vertex2_iterator   = map_key_iterator<vertex2_data_map>;
    using neighbor1_iterator = map_key_iterator<neighbor1_map>;
    using neighbor2_iterator = map_key_iterator<neighbor2_map>;
    using in1_edge_iterator  =
      map_bind2_iterator<neighbor1_map, edge_type, Vertex1>;
    using in2_edge_iterator  =
      map_bind2_iterator<neighbor2_map, edge_type, Vertex2>;
    using out1_edge_iterator =
      map_bind1_iterator<neighbor1_map, edge_type, Vertex1>;
    using out2_edge_iterator =
      map_bind1_iterator<neighbor2_map, edge_type, Vertex2>;
    using edge_iterator =
      map_map_iterator<vertex1_data_map, neighbor1_map, edge_type>;

    // Constructors, destructors, and related functions
    //--------------------------------------------------------------------------
  public:
    //! Creates an empty graph.
    bipartite_graph()
      : num_edges_(0) { }

    //! Creates a graph from a range of vertex pairs
    template <typename Range>
    explicit bipartite_graph(const Range& edges, typename Range::iterator* = 0)
      : num_edges_(0) {
      for (std::pair<Vertex1, Vertex2> p : edges) {
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
      num_edges_ = g.num_edges_;
      for (edge_type e : edges()) {
        EdgeProperty* ptr =
          new EdgeProperty(*static_cast<EdgeProperty*>(e.property_));
        data1_[e.v1()].neighbors[e.v2()] = ptr;
        data2_[e.v2()].neighbors[e.v1()] = ptr;
      }
      return *this;
    }

    //! Swap with another graph in constant time
    friend void swap(bipartite_graph& a, bipartite_graph& b) {
      swap(a.data1_, b.data1_);
      swap(a.data2_, b.data2_);
      std::swap(a.num_edges_, b.num_edges_);
    }

    // Accessors
    //--------------------------------------------------------------------------
  public:
    //! Returns the range of all type-1 vertices.
    iterator_range<vertex1_iterator>
    vertices1() const {
      return { data1_.begin(), data1_.end() };
    }

    //! Returns the range of all type-2 vertices.
    iterator_range<vertex2_iterator>
    vertices2() const {
      return { data2_.begin(), data2_.end() };
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { { data1_.begin(), data1_.end(), &vertex_data1_map::neighbors },
               { data1_.end(), data1_.end(), &vertex_data1_map::neighbors } };
    }

    //! Returns the type-2 vertices adjacent to a type-1 vertex.
    iterator_range<neighbor1_iterator>
    neighbors(Vertex1 u) const {
      const vertex1_data& data = data1_.at(u);
      return { data.neighbors.begin(), data.neighbors.end() };
    }

    //! Returns the type-1 vertices adjacent to a type-2 vertex.
    iterator_range<neighbor2_iterator>
    neighbors(Vertex2 u) const {
      const vertex2_data& data = data2_.at(u);
      return { data.neighbors.begin(), data.neighbors.end() };
    }

    //! Returns the edges incoming to a type-1 vertex.
    iterator_range<in1_edge_iterator>
    in_edges(Vertex1 u) const {
      const vertex1_data& data = data1_.at(u);
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns the edges incoming to a type-2 vertex.
    iterator_range<in2_edge_iterator>
    in_edges(Vertex2 u) const {
      const vertex2_data& data = data2_.at(u);
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns the edges outgoing from a type-1 vertex.
    iterator_range<out1_edge_iterator>
    out_edges(Vertex1 u) const {
      const vertex1_data& data = data1_.at(u);
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns the edges outgoing from a type-2 vertex.
    iterator_range<out2_edge_iterator>
    out_edges(Vertex2 u) const {
      const vertex2_data& data = data2_.at(u);
      return { { data.neighbors.begin(), u }, { data.neighbors.end(), u } };
    }

    //! Returns true if the graph contains the given type-1 vertex.
    bool contains(Vertex1 u) const {
      return data1_.find(u) != data1_.end();
    }

    //! Returns true if the graph contains the given type-2 vertex.
    bool contains(Vertex2 u) const {
      return data2_.find(u) != data2_.end();
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(Vertex1 u, Vertex2 v) const {
      auto it = data1_.find(u);
      return it != data1_.end() && it->second.neighbors.count(v);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(const edge_type& e) const {
      return contains(e.v1(), e.v2());
    }

    //! Returns an undirected edge between u and v. The edge must exist.
    bipartite_edge<Vertex1, Vertex2> edge(Vertex1 u, Vertex2 v) const {
      return { u, v, data1_.at(u).neighbors.at(v) };
    }

    //! Returns an undirected edge between u and v. The edge must exist.
    bipartite_edge<Vertex1, Vertex2> edge(Vertex2 u, Vertex1 v) const {
      return { u, v, data2_.at(u).neighbors.at(v) };
    }

    //! Returns the number of edges connected to a type-1 vertex.
    std::size_t degree(Vertex1 u) const {
      return data1_.at(u).neighbors.size();
    }

    //! Returns the number of edges connected to a type-2 vertex.
    std::size_t degree(Vertex2 u) const {
      return data2_.at(u).neighbors.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data1_.empty() && data2_.empty();
    }

    //! Returns the number of type-1 vertices.
    std::size_t num_vertices1() const {
      return data1_.size();
    }

    //! Returns the number of type-2 vertices.
    std::size_t num_vertices2() const {
      return data2_.size();
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return data1_.size() + data2_.size();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      return num_edges_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    edge_type reverse(const edge_type& e) const {
      return e.reverse();
    }

    //! Returns the property associated with a type-1 vertex.
    const Vertex1Property& operator[](Vertex1 u) const {
      return data1_.at(u).property;
    }

    //! Returns the property associated with a type-2 vertex.
    const Vertex2Property& operator[](Vertex2 u) const {
      return data2_.at(u).property;
    }

    //! Returns the property associated with a type-1 vertex.
    Vertex1Property& operator[](Vertex1 u) {
      return data1_.at(u).property;
    }

    //! Returns the property associated with a type-2 vertex.
    Vertex2Property& operator[](Vertex2 u) {
      return data2_.at(u).property;
    }

    //! Returns the property associated with an edge.
    const EdgeProperty& operator[](bipartite_edge<Vertex1, Vertex2> e) const {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](bipartite_edge<Vertex1, Vertex2> e) {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge must exist.
     */
    const EdgeProperty& operator()(Vertex1 u, Vertex2 v) const {
      return data1_.at(u).neighbors.at(v);
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge is added if necessary.
     */
    EdgeProperty& operator()(Vertex1 u, Vertex2 v) {
      return operator[](add_edge(u, v).first);
    }

    /**
     * Draws a random type-1 vertex in the graph, assuming one exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    Vertex1 sample_vertex1(Generator& rng) const {
      return sample_key(data1_, rng); // in sampling.hpp
    }

    /**
     * Draws a random type-2 vertex in the graph, assuming one exists.
     * \todo ensure sampling is uniform
     */
    template <typename Generator>
    Vertex2 sample_vertex2(Generator& rng) const {
      return sample_key(data2_, rng); // in sampling.hpp
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
        const neighbor1_map& neighbors = vit->second.neighbors;
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

    //! Compares the graph structure and the vertex & edge properties.
    bool operator!=(const bipartite_graph& g) const {
      return !operator==(g);
    }

    //! Prints the graph to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const bipartite_graph& g) {
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

    // Modifications
    //--------------------------------------------------------------------------
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
    std::pair<bipartite_edge<Vertex1, Vertex2>, bool>
    add_edge(Vertex1 u, Vertex2 v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != Vertex1() && v != Vertex2());
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
      ++num_edges_;
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
      vertex1_data& data = data1_.at(u);
      auto it = data.neighbors.find(v);
      assert(it != data.neighbors.end());

      // delete the edge data and the two symmetric edges
      delete it->second;
      data.neighbors.erase(it);
      data2_[v].neighbors.erase(u);
      --num_edges_;
    }

    //! Removes all edges incident to a type-1 vertex.
    void remove_edges(Vertex1 u) {
      neighbor1_map& neighbors = data1_.at(u).neighbors;
      num_edges_ -= neighbors.size();
      for (const auto& p : neighbors) {
        data2_[p.first].neighbors.erase(u);
        delete p.second;
      }
      neighbors.clear();
    }

    //! Removes all edges incident to a type-2 vertex.
    void remove_edges(Vertex2 u) {
      neighbor2_map& neighbors = data2_.at(u).neighbors;
      num_edges_ -= neighbors.size();
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
      num_edges_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data1_.clear();
      data2_.clear();
      num_edges_ = 0;
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
      std::size_t num_vertices1, num_vertices2, num_edges;
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

    //! The properties and neighbors of type-1 vertices.
    vertex1_data_map data1_;

    //! The properties and neighbors of type-2 vertices.
    vertex2_data_map data2_;

    //! The number of edges.
    std::size_t num_edges_;

  }; // class bipartite_graph

} // namespace libgm

#endif
