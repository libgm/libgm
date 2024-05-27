#ifndef LIBGM_CLUSTER_GRAPH_HPP
#define LIBGM_CLUSTER_GRAPH_HPP

#include <libgm/argument/annotated.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/set_index.hpp>
#include <libgm/graph/algorithm/mst.hpp>
#include <libgm/graph/algorithm/test_connected.hpp>
#include <libgm/graph/algorithm/test_tree.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/algorithm/triangulate.hpp>
#include <libgm/graph/util/bidirectional.hpp>
#include <libgm/graph/util/id.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/graph/util/void.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace libgm {

/**
 * Represents a cluster graph \f$G\f$. Each vertex and edge of the graph
 * is associated with a domain, called a cluster and separator, respectively.
 * Each vertex and each edge can also be associated with a user-specified
 * property. The graph is undirected, but if needed, information for both
 * directions of the edge can be stored using the bidirectional class.
 *
 * \ingroup graph_types
 */
class ClusterGraph : public Object {

  // Public type declarations
  //--------------------------------------------------------------------------
public:
  /**
   * The information stored with each vertex of the cluster graph.
   */
  class Vertex {
  public:
    using NeighborMap = ankerl::unordered_dense::map<Vertex*, Edge*, Hash>;

    /// Default constructor. Default-initializes the property.
    Vertex() = default;

    /// Constructs the vertex info with the given cluster and property.
    Vertex(Domain cluster, Object object)
      : cluster(std::move(cluster)), property(std::move(property)) { }

    const Domain& cluster() {
      return cluster_;
    }

    size_t index() const {
      return index_;
    }

    Shape shape(const ShapeMap& map, std::vector<size_t>& vec) const {
      return cluster_.shape(map, vec);
    }

    void save(oarchive& ar) const;
    void load(iarchive& ar);
    friend std::ostream& operator<<(std::ostream& out, const Vertex& v);

  private:
    /// The cluster of variables.
    Domain cluster_;

    /// The vertex property.
    Object property_;

    /// True if the vertex has been marked. This field is not serialized.
    bool marked_ = false;

    /// The index of the vertex in the graph vector, useful for vertex_index map.
    size_t index_ = -1;

    /// The hash.
    size_t hash_ = 0;

    /// The set of neighbours and corresponding edge data.
    NeighborMap neighbors_;

  }; // struct Vertex

  /**
   * The information stored with each edge of the cluster graph.
   */
  class Edge {
  public:
    /// Default constructor. Default-initalizes the property.
    Edge()
      : property(), marked(false) { }

    /// Constructs the edge info with the given separator and property.
    Edge(Domain separator, PImplObject property)
      : separator(std::move(separator)), property(std::move(property)) { }

    const Domain& separator() const {
      return separator_;
    }

    Shape shape(const ShapeMap& map, std::vector<size_t>& vec) const {
      return separator_.shape(map, vec);
    }

    /// Compares the separators and properties stored at two edges.
    friend bool operator==(const Edge& a, const Edge& b) {
      return a.separator == b.separator && a.property == b.property;
    }

    /// Compares the separators and properties stored at two edges.
    friend bool operator!=(const Edge& a, const Edge& b) {
      return !(a == b);
    }

    /// Serialize members.
    template <typename Archive>
    void save(Archive& ar, unsigned version) const {
      ar << separator << property;
    }

    /// Deserialize members
    template <typename Archive>
    void load(Archive& ar, unsigned version) {
      ar >> separator >> property;
    }

    /// Outputs the edge information to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const Edge& info) {
      out << '(' << info.separator << ' ' << info.property << ' ' << info.marked << ')';
      return out;
    }

  private:
    /// The separator.
    Domain separator_;

    /// The edge property annotated with the separator.
    Object property_;

    /// True if the edge has been marked. This field is not serialized.
    bool marked_ = false;

    /**
     * For edge = (u, v), reachable.directed(e) stores the variables
     * in the subtree rooted at u, away from v, in the sorted order.
     * This field is not serialized.
     */
    Bidirectional<Domain> reachable_;

  }; // class Edge

  // Vertex type, edge type, argument_type, and properties
  using vertex_descriptor = Vertex*;
  using edge_descriptor   = UndirectedEdge<Vertex*, Edge>;
  // using vertex_property = VertexProperty;
  // using edge_property   = EdgeProperty;

  // Iterators
  using adjacency_iterator = Vertex::adjacency_iterator;
  using neighbor_iterator = typename graph_type::neighbor_iterator;
  using out_edge_iterator = typename graph_type::out_edge_iterator;
  using in_edge_iterator = typename graph_type::in_edge_iterator;
  using vertex_iterator = std::vector<Vertex*>::const_iterator;
  using edge_iterator = typename graph_type::edge_iterator;

    using adjacency_iterator = MapKeyIterator<NeighborMap>;
    using out_edge_iterator  = MapBind1Iterator<NeighborMap, edge_descriptor>;
    using in_edge_iterator   = MapBind2Iterator<NeighborMap, edge_descriptor>;



  using argument_iterator = /* ???? */
    typename set_index<Vertex*, Domain >::value_iterator;

  // Visitors
  using vertex_visitor = std::function<void(vertex_descriptor)>;
  using edge_visitor = std::function<void(edge_descriptor)>;

  // Constructors and destructors
  //--------------------------------------------------------------------------
public:
  /// Constructs an empty cluster graph with no clusters.
  ClusterGraph()
    : max_id_(0) { }

  /* private */
  /// Swaps two cluster graphs in constant time.
  friend void swap(ClusterGraph& a, ClusterGraph& b) {
    swap(a.cluster_index_, b.cluster_index_);
    swap(a.separator_index_, b.separator_index_);
    swap(a.graph_, b.graph_);
    std::swap(a.max_id_, b.max_id_);
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  /// Serializes the cluster graph to an archive.
  template <typename Archive>
  void save(Archive& ar, unsigned version) const {
    ar << data_;
  }

  /// Deserialize the cluster graph from an archive.
  template <typename Archive>
  void load(Archive& ar, unsigend version) {
    ar >> data_;
    after_load();
  }

  /// Prints a human-readable representation of the cluster graph to stream.
  friend std::ostream& operator<<(std::ostream& out, const ClusterGraph& g);

  static Vertex* null_vertex() {
    return nullptr;
  }

  // Graph concept
  //--------------------------------------------------------------------------
  /// Returns the vertices adjacent to u.
  friend IteratorPair<adjacency_iterator> adjacent_vertices(Vertex* u, const ClusterGraph& g);

  /// Returns the outgoing edges from a vertex.
  friend IteratorPair<out_edge_iterator> out_edges(Vertex* u, const ClusterGraph& g);

  /// Returns the edges incoming to a vertex.
  friend IteratorPair<in_edge_iterator> in_edges(Vertex* u, const ClusterGraph& g);

  /// Returns the range of all vertices.
  friend IteratorPair<vertex_iterator> vertices(const ClusterGraph& g);

  /// Returns all edges in the graph.
  friend IteratorPair<edge_iterator> edges(const ClusterGraph& g);

  // Graph accessors
  //--------------------------------------------------------------------------

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the pointer to the first vertex.
  vertex_iterator begin() const;

  /// Returns the pointer past the last vertex.
  vertex_iterator end() const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Vertex* u) const;

  /// Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Vertex* u, Vertex* v) const;

  /// Returns true if the graph contains an undirected edge.
  bool contains(UndirectedEdge<Vertex*> e) const;

  /// Returns the union of all the clusters in this graph.
  boost::iterator_range<argument_iterator> arguments() const;

  /// Returns the cardinality of the union of all the clusters.
  size_t num_arguments() const;

  /// Returns true if the number of clusters in which the argument occurs.
  size_t count(Arg x) const;

  /// Returns the property associated with a vertex.
  Object& operator[](Vertex* u);

  /// Returns the property associated with a vertex.
  const Object& operator[](Vertex* u) const;

  /// Returns the property associated with an edge.
  Object& operator[](UndirectedEdge<Vertex*> e);

  /// Returns the property associated with an edge
  const Object& operator[](UndirectedEdge<Vertex*> e) const;

  // Returns the property associated with edge {u, v}. The edge must exist.
  Object& operator()(Vertex* u, Vertex* v);

  // Returns the property associated with edge {u, v}. The edge must exist.
  const Object& operator()(Vertex* u, Vertex* v) const;

  /// Returns true if two cluster graphs are identical.
  bool operator==(const ClusterGraph& other) const;

  /// Returns true if two cluster graphs are not identical.
  bool operator!=(const ClusterGraph& other) const;

  // Queries
  //--------------------------------------------------------------------------

  /**
   * Returns true if the graph is connected.
   */
  bool is_connected() const;

  /**
   * Returns true if the cluster graph is a tree.
   */
  bool is_tree() const;

  /**
   * Returns true if the cluster graph satisfies the running intersection
   * property. A cluster graph satisfies the running intersection property
   * if, for each value, the clusters and separators containing that value
   * form a subtree.
   */
  bool has_running_intersection() const;

  /**
   * Returns true if the cluster graph represents a triangulated model,
   * i.e., is a tree and satisfies the running intersection property.
   */
  bool is_triangulated() const;

  /**
   * Returns the maximum clique size minus one.
   * Only meaningful when this graph is a tree.
   */
  int tree_width() const;

  /**
   * Returns a vertex whose clique covers (is a superset of) the supplied
   * domain. If there are multiple such vertices, returns the one with the
   * smallest cluster size (cardinality). If there is no such vertex,
   * then returns the null vertex.
   */
  vertex_descriptor find_cluster_cover(const Domain& dom) const;

  /**
   * Returns an edge whose separator covers the supplied domain.
   * If there are multiple such edges, one with the smallest separator
   * is returned. If there is no such edge, returns a null edge.
   */
  edge_descriptor find_separator_cover(const Domain& dom) const;

  /**
   * Returns a vertex whose clique cover meets (intersects) the supplied
   * domain. The returned vertex is the one that has the smallest cluster
   * that has maximal intersection with the supplied domain.
   */
  vertex_descriptor find_cluster_meets(const Domain& dom) const;

  /**
   * Returns an edge whose separator meets (intersects) the supplied
   * domain. The returned edge is the one that has the smallest separator
   * that has maximal intersection with the supplied domain.
   */
  edge_descriptor find_separator_meets(const Domain& dom) const;

  /**
   * Visits the vertices whose clusters overlap the supplied domain.
   */
  void intersecting_clusters(const Domain& dom, vertex_visitor visitor) const;

  /**
   * Visits the edges whose separators overlap the supplied domain.
   */
  void intersecting_separators(const Domain& dom, edge_visitor visitor) const;

  /**
   * Computes the reachable nodes for each directed edge in the
   * cluster graph. Requires the graph to be a tree.
   *
   * \param propagate_past_empty
   *        If set to false, then edges with empty separators
   *        are ignored. Their reachable sets are assigned empty sets.
   */
  void compute_reachable(bool past_empty);

  /**
   * Computes the reachable nodes for each directed edge in the
   * junction tree.
   *
   * \param propagate_past_empty
   *        If set to false, then edges with empty separators
   *        are ignored. Their reachable sets are assigned empty sets.
   * \param filter
   *        The reachable node sets are intersected with this set.
   */
  void compute_reachable(bool past_empty, const Domain& filter);

  /**
   * Marks the smallest subtree (or subforest) of this junction tree
   * whose cliques cover the supplied set of nodes.
   *
   * When the junction tree has empty separators, it can be regarded
   * as a forest of independent junction trees.  In this case, a set
   * of variables may be covered by a non-contiguous subgraph of the
   * total junction tree.  The argument force_continuous controls
   * whether whether the function marks a connected or is allowed to
   * mark a set of disconnected subtree covers.
   *
   * \param domain
   *        the nodes to be covered
   * \param force_continuous
   *        if true, the function is guaranteed to mark a connected subgraph.
   */
  void mark_subtree_cover(const Domain& dom, bool force_continuous);

  // Mutating operations
  //--------------------------------------------------------------------------

  /**
   * Adds a vertex with the given cluster and vertex property.
   */
  Vertex* add_vertex(Domain cluster, Object vp = Object()) {

  /**
   * Adds an edge {u, v} to the graph with the given separator. The edge
   * endpoints must exist, and the separator must be a subset of the clusters
   * at these vertices. If the edge already exists, does not perform anything.
   * \return the edge and bool indicating whether the insertion took place
   */
  std::pair<UndirectedEdge<Vertex*>, bool>
  add_edge(Vertex* u, Vertex* v, Domain separator, Object ep = Object());

  /**
   * Adds an edge {u, v} to the graph, setting the separator to the
   * intersection of the two clusters at the endpoints.
   * If the edge already exists, does not perform anything.
   * \return the edge and bool indicatign whether the insertion took place
   */
  std::pair<UndirectedEdge<Vertex*>, bool> add_edge(Vertex* u, Vertex* v);

  /**
   * Updates the cluster associated with an existing vertex.
   */
  void update_cluster(Vertex* u, const Domain& cluster);

  /**
   * Updates the separator associated with an edge.
   */
  void update_separator(UndirectedEdge<Vertex*> e, const Domain& separator);

  /**
   * Merges two adjacent vertices and their clusters. The edge (u,v) and
   * the source vertex u are deleted, and the target vertex v is made
   * adjacent to all neighbors of u (the information on these edges is
   * copied from the original edges). The new clique of v is set as the
   * union of the original cliques of v and u, and the separators remain
   * unchanged. If the graph is a tree that satisfies the running
   * intersection property, the property will still hold after the merge.
   *
   * \return the retained vertex
   */
  Vertex* merge(UndirectedEdge<Vertex*> e);

  /// Removes a vertex and the associated cluster and property.
  void remove_vertex(Vertex* v);

  /// Removes an undirected edge {u, v} and the associated separator and data.
  void remove_edge(Vertex* u, Vertex* v);

  /// Removes all edges incindent to a vertex
  void remove_edges(Vertex* u);

  /// Removes all edges from the graph
  void remove_edges();

  /// Removes all vertices and edges from the graph
  void clear();

  // Triangulation
  //--------------------------------------------------------------------------

  /**
   * Extends the cliques, so that tree satisfies the running intersection
   * property.
   */
  void triangulate();

  /**
   * Initializes this cluster graph to a junction tree that represents the
   * triangulation of the supplied graph.
   *
   * \tparam Graph an undirected graph type whose vertices correspond to
   *               variables stored in Domain.
   * \tparam Strategy a type that models the EliminationStrategy concept
   */
  void triangulated(MarkovNetwork& mn, Strategy& strategy) {
    clear();
    libgm::triangulate_maximal<Domain>(g, [&](Domain&& dom) {
        add_cluster(dom); }, strategy);
    mst_edges();
  }

  /**
   * Initializes the cluster graph to a junction tree with the given
   * cliques of a triangulated greaph.
   *
   * \tparam Range a range with values convertible to Domain
   */
  void triangulated(const std::vector<Domain>& cliques);

  /**
   * Initializes the edges using the maximum spanning tree algorithm based
   * on the separator sizes associated with each edge. This guarantees that
   * if the clusters are triangulated to begin with, the cluster graph will
   * be a junction tree.
   */
  void mst_edges();

}; // class ClusterGraph




  /// Returns an undirected edge (u, v). The edge must exist.
  UndirectedEdge<Vertex*> edge(Vertex* u, Vertex* v);

  /// Returns the number of edges adjacent to a vertex.
  std::size_t in_degree(Vertex* u) const;

  /// Returns the number of edges adjacent to a vertex.
  std::size_t out_degree(Vertex* u) const;

  /// Returns the number of edges adjacent to a vertex.
  std::size_t degree(Vertex* u) const;

  /// Returns the number of vertices
  std::size_t num_vertices() const;

  /// Returns the number of edges
  std::size_t num_edges() const;



template <typename VertexProperty>
struct ClusterGraphT1 : ClusterGraph {
  VertexProperty& operator[](Vertex* u) {
    return static_cast<VertexProperty&>(ClusterGraph::operator[](u));
  }

  /// Returns the strongly-typed property associated with a vertex.
  const VertexProperty& operator[](Vertex* u) const {
    return static_cast<const VertexProperty&>(ClusterGraph::operator[](u));
  }

  Vertex* add_vertex(Domain cluster, VertexProperty vp = VertexProperty()) {
    return ClusterGraph::add_vertex(std::move(cluster), std::move(vp));
  };
};

template <>
struct ClusterGraphT1 : ClusterGraph {
  void operator[](Vertex* u) {}
  void operator[](Vertex* u) const {}
  Vertex* add_vertex(Domain cluster) { return ClusterGraph::add_vertex(std::move(cluster));}
};

/**
 * \tparam VertexProperty
 *         A type of values stored at each vertex.
 *         Must be DefaultConstructible and the CopyConstructible.
 * \tparam EdgeProperty
 *         A type of values stored at each edge.
 *         Must be DefaultConstructible and the CopyConstructible concept.
 *         should have a fast default constructor, since we temporarily
 *         create a superlinear number of edges to compute the MST.
 *
 */
template <typename VertexProperty = void, typename EdgeProperty = void>
struct ClusterGraphT : ClusterGraphT1<VertexProperty> {
  EdgeProperty& operator[](const edge_descriptor& e) {
    return static_cast<EdgeProperty&>(ClusterGraph::operator[](e));
  }

  const EdgeProperty& operator[](const edge_descriptor& e) const {
    return static_cast<const EdgeProperty&>(ClusterGraph::operator[](e));
  }

  EdgeProperty& operator()(Vertex* u, Vertex* v) {
    return static_cast<EdgeProperty&>(ClusterGraph::operator()(u, v));
  }

  const EdgeProperty& operator()(Vertex* u, Vertex* v) const {
    return static_cast<const EdgeProperty&>(ClusterGraph::operator()(u, v));
  }

  std::pair<edge_descriptor, bool>
  add_edge(Vertex* u, Vertex* v, Domain separator, EdgeProperty ep = EdgeProperty()) {
    return ClusterGraph::add_edge(u, v, std::move(separator), std::move(ep));
  }
};

template <typename VertexProperty>
struct ClusterGraphT<VertexProperty, void> : ClusterGraphT1<VertexProperty> {
  void operator[](const edge_descriptor& e) {}
  void operator[](const edge_descriptor& e) const {}
  void operator()(Vertex* u, Vertex* v) {}
  void operator()(Vertex* u, Vertex* v) const {}
  std::pair<edge_descriptor, bool> add_edge(Vertex* u, Vertex* v, Domain seperator) {
    return ClusterGRaph::add_edge(u, v, std::move(separator));
  }
};

} // namespace libgm

#endif
