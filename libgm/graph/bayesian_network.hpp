#ifndef LIBGM_BAYESIAN_NETWORK_HPP
#define LIBGM_BAYESIAN_NETWORK_HPP

#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/math/logarithmic.hpp>

#include <iterator>
#include <unordered_map>

namespace libgm {

  /**
   * A graph that represents a Bayesian network.
   *
   * A Bayesian network is a directed acyclic graph, where each vertex is an
   * argument. Each vertex is associated with a domain; the elements of the
   * domain determine the parents of the vertex.
   *
   * Each vertex is associated with a property, but edges are not.
   *
   * \ingroup model
   */
  template <typename Arg, typename VertexProperty>
  class bayesian_network {

    // Private types
    //--------------------------------------------------------------------------
  private:
    //! The map type used to associate neighbors and edge data with each vertex.
    using neighbor_set = std::unordered_set<Arg>;

    /**
     * A struct with the data associated with each vertex. This structure
     * stores the property associated with the vertex as well all edges from/to
     * parent and child vertices (along with the edge properties).
     */
    struct vertex_data {
      annotated<Arg, VertexProperty> property;
      neighbor_set children;
      bool operator==(const vertex1_data& other) const {
        return property == other.property; // intentionally omitting neihgbors
      }
    };

    //! The map types that associates all the vertices with their vertex_data.
    using vertex_data_map = std::unordered_map<Arg, vertex_data>;

    //--------------------------------------------------------------------------
  public:
    // Vertex types, edge type, and properties
    using argument_type   = Arg;
    using vertex_type     = Arg;
    using edge_type       = directed_edge<Arg>;
    using vertex_property = VertexProperty;
    using edge_property   = void;

    // Iterators
    using argument_iterator = map_key_iterator<vertex_data_map>;
    using vertex_iterator   = map_key_iterator<vertex_data_map>;
    using parent_iterator   = typename domain<Arg>::const_iterator;
    using child_iterator    = typename neighbor_set::const_iterator;
    using in_edge_iterator  = bind2_iterator<domain<Arg>, edge_type>;
    using out_edge_iterator = bind1_iterator<neighbor_set, edge_type>;
    using edge_iterator =
      map_range_iterator<vertex_data_map, neighbor_set, edge_type>;
    using iterator = map_value_property_iterator<vertex_data_map>;

    // Constructors
    //--------------------------------------------------------------------------
  public:
    //! Default constructor. Creates an empty Bayesian network.
    bayesian_network()
      : num_edges_(0) { }

    //! Swaps two Bayesian networks in constant time.
    friend void swap(bayesian_network& a, bayesian_network& b) {
      using std::swap;
      swap(a.data_, b.data_);
      swap(a.num_edges_, b.num_edges_);
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the range of all arguments (same as vertices()).
    iterator_range<argument_iterator>
    arguments() const {
      return { data_.begin(), data_.end() };
    }

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
    iterator_range<parent_iterator>
    parents(Arg u) const {
      const domain<Arg>& dom = data_.at(u).property.domain;
      return { ++dom.begin(), dom.end() };
    }

    //! Returns the children of u.
    iterator_range<child_iterator>
    children(Arg u) const {
      const vertex_data& data = data_.at(u);
      return { data.children.begin(), data.children.end() };
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(Arg u) const {
      iterator_range<parent_iterator> range = parents(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(Arg u) const {
      iterator_range<child_iterator> range = children(u);
      return { { range.begin(), u }, { range.end(), u } };
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(Arg u) const {
      return data_.find(u) != data_.end();
    }

    //! Returns true if the graph contains a directed edge (u, v).
    bool contains(Arg u, Arg v) const {
      auto it = data_.find(u);
      return it != data_.end() && it->second.children.count(v);
    }

    //! Returns true if the graph contains a directed edge.
    bool contains(const directed_edge<Arg> e) const {
      return contains(e.source(), e.target());
    }

    //! Returns a directed edge (u,v) between two vertices. The edge must exist.
    directed_edge<Arg> edge(Arg u, Arg v) const {
      return { u, v };
    }

    /**
     * Returns the number of incoming edges to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t in_degree(Arg u) const {
      return data_.at(u).property.domain.size() - 1;
    }

    /**
     * Returns the number of outgoing edges to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t out_degree(Arg u) const {
      return data_.at(u).children.size();
    }

    /**
     * Returns the total number of edges adjacent to a vertex. The vertex must
     * already be present in the graph.
     */
    std::size_t degree(Arg u) const {
      const vertex_data& data = data_.at(u);
      return data.property.domain.size() - 1 + data.children.size();
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return data_.empty();
    }

    //! Returns the number of arguments (as as num_vertices()).
    std::size_t num_arguments() const {
      return data_.size();
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return data_.size();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      return num_edges_;
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator[](Arg u) const {
      return data_.at(u).property;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](Arg u) {
      return data_.at(u).property;
    }

    //! Returns the arguments of a factor.
    const domain<Arg>& arguments(Arg u) const {
      return data_.at(u).property.domain;
    }

    //! Returns the annotated property associated with an argument.
    const annotated<Arg, ArgumentProperty>& property(Arg u) const {
      return data_.at(u).property;
    }

    //! Returns the begin iterator over the range of all properties.
    iterator begin() const {
      return data_.begin();
    }

    //! Returns the end iterator over the range of all properties.
    iterator end() const {
      return data_.end();
    }

    /**
     * Compares the graph strucutre and the vertex & edge properties.
     * The property types must support operator!=().
     */
    bool operator==(const bayesian_network& other) const {
      return
        num_vertices() == g.num_vertices() &&
        num_edges() == g.num_edges() &&
        data_ == g.data_;
    }

    //! Inequality comparison
    bool operator!=(const bayesian_network& other) const {
      return !(*this == other);
    }

    //! Prints a directed to an output stream.
    friend std::ostream&
    operator<<(std::ostream& out, const bayesian_network& g) {
      for (Arg u : g.vertices()) {
        out << u << ": " << g.property(u) << std::endl;
      }
      return out;
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Computes a minimal Markov graph capturing dependencies in this model.
     */
    void markov_graph(undirected_graph<Arg>& mg) const {
      for (Arg v : vertices()) {
        mg.make_clique(arguments(v));
      }
    }

    // Modifications
    //--------------------------------------------------------------------------

    /**
     * Adds an argument to the graph with the given domain and associates
     * a property with that vertex. The domain must not be empty.
     * If the vertex is already present, its property and edges are overwritten.
     *
     * \returns true if the vertex was newly inserted
     */
    bool add_vertex(const domain<Arg>& dom,
                    const VertexProperty& p = VertexProperty()) {
      assert(!dom.empty());
      Arg u = dom.front();
      auto it = data_.find(u);
      bool existing_vertex = it != data_.end();
      if (existing_vertex) {
        remove_in_edges(u);
      }
      vertex_data& data = existing_vertex ? it->second : data_[u];
      data.property.domain = dom;
      data.property.object = p;
      for (Arg v : make_iterator_range(++dom.begin(), dom.end())) {
        assert(u != v);
        data_[v].children.insert(u);
      }
      num_edges_ += dom.size() - 1;
      return !existing_vertex;
    }

    //! Removes a vertex from the graph, provided that it has no outoing edges.
    void remove_vertex(Arg u) {
      auto it = data_.find(u);
      assert(it != data_.end() && it->second.children.empty());
      remove_in_edges(u);
      data_.erase(it);
    }

    //! Removes all edges incoming to a vertex.
    void remove_in_edges(Arg u) {
      domain<Arg>& dom = data_.at(u).property.domain;
      num_edges_ -= dom.size() - 1;
      for (Arg v : dom) {
        data_[v].erase(u);
      }
      dom.resize(1);
    }

    //! Removes all vertices and edges from the graph.
    void clear() {
      data_.clear();
      num_edges_ = 0;
    }

    //! Saves the graph to an archive.
    void save(oarchive& ar) const {
      ar << num_vertices() << num_edges_;
      for (const auto& p : data_) {
        ar << p.first << p.second.property;
      }
    }

    //! Loads the graph from an archive.
    void load(iarchive& ar) {
      clear();
      std::size_t num_vertices;
      Arg u;
      ar >> num_vertices;
      ar >> num_edges_;
      while (num_vertices-- > 0) {
        ar >> u;
        ar >> data_[u].property;
        const domain<Arg>& dom = data_[u].property.domain;
        for (Arg v : make_iterator_range(++dom.begin(), dom.end())) {
          assert(u != v);
          data_[v].children.insert(u);
        }
      }
    }

  }; // class bayesian_network

} // namespace libgm

#endif
