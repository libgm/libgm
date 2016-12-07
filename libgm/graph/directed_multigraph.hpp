#ifndef LIBGM_DIRECTED_MULTIGRAPH_HPP
#define LIBGM_DIRECTED_MULTIGRAPH_HPP

#include <libgm/datastructure/vector_map.hpp>
#include <libgm/graph/directed_edge.hpp>
#include <libgm/graph/util/void.hpp>
#include <libgm/iterator/map_bind1_iterator.hpp>
#include <libgm/iterator/map_bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/iterator/map_map_iterator.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

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
   * \tparam Vertex
   *         The type that represents a vertex.
   * \tparam VertexProperty
   *         The type of data associated with vertices.
   * \tparam EdgeProperty
   *         The type of data associated with edges.
   *
   * \ingroup graph_types
   * \see Graph
   */
  template <typename Vertex,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class directed_multigraph {

    // Private types
    //--------------------------------------------------------------------------
  private:
    //! The map type used to associate neighbors and edge data with each vertex.
    using neighbor_map = vector_map<Vertex, EdgeProperty*>;

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
    using vertex_data_map = std::unordered_map<Vertex, vertex_data>;

    // Public type declerations
    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge type, and properties
    using vertex_type     = Vertex;
    using edge_type       = directed_edge<Vertex>;
    using vertex_property = VertexProperty;
    using edge_property   = EdgeProperty;

    // Iterators
    using vertex_iterator   = map_key_iterator<vertex_data_map>;
    using neighbor_iterator = map_key_iterator<neighbor_map>;
    using in_edge_iterator  = map_bind2_iterator<neighbor_map, edge_type>;
    class out_edge_iterator = map_bind1_iterator<neighbor_map, edge_type>;
    using edge_iterator =
      map_map_iterator<vertex_data_map, neighbor_map, edge_type>;

    // Constructors and destructors
    //--------------------------------------------------------------------------
  public:
    //! Create an empty graph.
    directed_multigraph()
      : num_edges_(0) { }

    //! Create a graph from a list of pairs.
    template <typename Range>
    explicit directed_multigraph(const Range& edges,
                                 typename Range::iterator* = 0)
      : num_edges_(0) {
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
      if (this == &other) { return *this; }
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
      std::swap(a.num_edges_, b.num_edges_);
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return { data_.begin(), data_.end() };
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return { { data_.begin(), data_.end(), &vertex_data_map::children },
               { data_.end(), data_.end(), &vertex_data_map::children } };
    }

    //! Returns the parents of u.
    iterator_range<neighbor_iterator>
    parents(Vertex u) const {
      const vertex_data& data = data_.at(u);
      return { { data.parents.begin(), { data.parents.end() } };
    }

    //! Returns the children of u.
    iterator_range<neighbor_iterator>
    children(Vertex u) const {
      const vertex_data& data = data_.at(u);
      return { data.children.begin(), data.children.end() };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(Vertex u) const {
      const vertex_data& data = data_.at(u);
      return { { data.parents.begin(), u },  { data.parents.end(), u } };
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(Vertex u) const {
      const vertex_data& data = data_.at(u);
      return { { data.children.begin(), u }, { data.children.end(), u } };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(Vertex u) const {
      return data_.find(u) != data_.end();
    }

    //! Returns true if the graph contains a directed edge (u, v).
    bool contains(Vertex u, Vertex v) const {
      auto it = data_.find(u);
      return it != data_.end() && it->second.children.count(v);
    }

    //! Returns true if the graph contains a directed edge.
    bool contains(const edge_type& e) const {
      return contains(e.source(), e.target());
    }

    //! Returns a directed edge (u,v) between two vertices. The edge must exist.
    directed_edge<Vertex> edge(Vertex u, Vertex v) const {
      return { u, v, data_.at(u).children.at(v) };
    }

    /**
     * Returns the number of incoming edges to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t in_degree(Vertex u) const {
      return data_.at(u).parents.size();
    }

    /**
     * Returns the number of outgoing edges to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t out_degree(Vertex u) const {
      return data_.at(u).children.size();
    }

    /**
     * Returns the total number of edges adjacent to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t degree(Vertex u) const {
      const vertex_data& data = data_.at(u);
      return data.parents.size() + data.children.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data_.empty();
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return data_.size();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      return num_edges_;
    }

    //! Given a directed edge (u, v), returns a directed edge (v, u).
    //! The edge (v, u) must exist.
    directed_edge<Vertex> reverse(directed_edge<Vertex> e) const {
      return edge(e.target(), e.source());
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator[](Vertex u) const {
      return data_.at(u).property;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](Vertex u) {
      return data_.at(u).property;
    }

    //! Returns the property associated with an edge.
    const EdgeProperty& operator[](directed_edge<Vertex> e) const {
      return *static_cast<EdgeProperty*>(e.property_);
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](directed_edge<Vertex> e) {
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

    //! Print graphs to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const directed_multigraph& g) {
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

    // Modifications
    //--------------------------------------------------------------------------
    /**
     * Adds a vertex to the graph and associates a property with that vertex.
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
     * Adds an edge (u, v) to the graph.
     * If u and v are not present, they are added.
     * \return the edge and true (the edge is always added).
     */
    std::pair<edge_type, bool>
    add_edge(Vertex u, Vertex v, const EdgeProperty& p = EdgeProperty()) {
      assert(u != null_vertex());
      assert(v != null_vertex());
      EdgeProperty* ptr = new EdgeProperty(p);
      data_[u].children.emplace(v, ptr);
      data_[v].parents.emplace(u, ptr);
      ++num_edges_;
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
      vertex_data& data = data_.at(u);
      typename neighbor_map::iterator begin, end;
      std::tie(begin, end) = data.children.equal_range(v);
      num_edges_ -= std::distance(begin, end);

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
      neighbor_map& parents = data_.at(u).parents;
      num_edges_ -= parents.size();
      for (const auto& p : parents) {
        data_[p.first].children.erase(u);
        if (p.second) { delete p.second; }
      }
      parents.clear();
    }

    //! Removes all edges outgoing from a vertex.
    void remove_out_edges(Vertex u) {
      neighbor_map& children = data_.at(u).children;
      num_edges_ -= children.size();
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
      num_edges_ = 0;
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      free_edge_data();
      data_.clear();
      num_edges_ = 0;
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
      std::size_t num_vertices, num_edges;
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

    // Private members
    //--------------------------------------------------------------------------
  private:
    void free_edge_data() {
      for (edge_type e : edges()) {
        if (e.property_) {
          delete static_cast<EdgeProperty*>(e.property_);
        }
      }
    }

    //! A map from each vertex to its vertex data.
    vertex_data_map data_;

    //! The total number of directed edges in the graph.
    std::size_t num_edges_;

  }; // class directed_multigraph

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

    typedef std::size_t vertices_size_type;
    typedef std::size_t edges_size_type;
    typedef std::size_t degree_size_type;

    static vertex_descriptor null_vertex() {
      return Vertex();
    }
  };

} // namespace boost

#endif
