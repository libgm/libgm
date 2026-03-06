#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/domain_index.hpp>
#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/datastructure/subrange.hpp>
#include <libgm/graph/intrusive_edge.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/util/bgl.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/casting_iterator.hpp>
#include <libgm/iterator/member_iterator.hpp>
#include <libgm/opaque.hpp>

#include <ankerl/unordered_dense.h>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>

#include <cassert>
#include <memory>
#include <new>
#include <typeinfo>
#include <type_traits>
#include <utility>

namespace libgm {

/**
 * Represents a cluster graph \f$G\f$. Each vertex and edge of the graph
 * is associated with a domain, called a cluster and separator, respectively.
 * Each vertex and each edge can also be associated with a user-specified
 * property. The graph is undirected, but if needed, information for both
 * directions of the edge can be stored using the Bidirectional class.
 *
 * \ingroup graph_types
 */
class ClusterGraph {
public:
  // The implementation types
  struct Impl;
  Impl& impl();
  const Impl& impl() const;

  // The types of data stored in each vertex and edge.
  struct Vertex;
  struct Edge;

  // Vertex I/O
  friend std::ostream& operator<<(std::ostream& out, Vertex* v);

  // Public type declarations
  //--------------------------------------------------------------------------
public:
  // Descriptors
  using vertex_descriptor = Vertex*;
  using edge_descriptor = IntrusiveEdge<Vertex, Edge>;

  // Graph categories
  using directed_category = boost::directed_tag;
  using edge_parallel_category = boost::disallow_parallel_edge_tag;
  struct traversal_category :
    public virtual boost::vertex_list_graph_tag,
    public virtual boost::incidence_graph_tag,
    public virtual boost::adjacency_graph_tag,
    public virtual boost::edge_list_graph_tag { };

  // Size types
  using vertices_size_type = size_t;
  using edges_size_type = size_t;
  using degree_size_type = size_t;

  // Iterators (the exact types are implementation detail)
  using out_edge_iterator  = CastingIterator<IntrusiveList<Edge>::entry_iterator, edge_descriptor>;
  using in_edge_iterator   = MemberIterator<out_edge_iterator, &edge_descriptor::reverse>;
  using adjacency_iterator = MemberIterator<out_edge_iterator, &edge_descriptor::target>;
  using vertex_iterator    = IntrusiveList<Vertex>::iterator;
  using edge_iterator      = IntrusiveList<Edge>::iterator;
  using argument_iterator  = DomainIndex<Vertex>::argument_iterator;

  // Color map for vertices
  struct VertexColorMap {
    using value_type = boost::default_color_type;
    using reference = boost::default_color_type;
    using key_type = Vertex*;
    using category = boost::read_write_property_map_tag;
  };

  // Index map for vertices
  struct VertexIndexMap {
    using value_type = size_t;
    using reference = size_t;
    using key_type = Vertex*;
    using category = boost::readable_property_map_tag;
  };

  // Visitors
  using VertexVisitor = std::function<void(vertex_descriptor)>;
  using EdgeVisitor = std::function<void(edge_descriptor)>;

  // Constructors and destructors
  //--------------------------------------------------------------------------
protected:
  ClusterGraph(PropertyLayout vertex_layout, PropertyLayout edge_layout);

public:
  /// Constructs an empty cluster graph with no clusters.
  ClusterGraph();
  ClusterGraph(const ClusterGraph& other);
  ClusterGraph(ClusterGraph&& other) noexcept = default;
  ClusterGraph& operator=(const ClusterGraph& other);
  ClusterGraph& operator=(ClusterGraph&& other) noexcept = default;
  ~ClusterGraph();

  /// Swaps two cluster graphs in constant time.
  friend void swap(ClusterGraph& a, ClusterGraph& b) { std::swap(a.impl_, b.impl_); }

  static Vertex* null_vertex() { return nullptr; }

  VertexColorMap vertex_color_map() { return {}; }

  VertexIndexMap vertex_index_map() { return {}; }

  // Graph accessors
  //--------------------------------------------------------------------------

  /// Returns the outgoing edges from a vertex.
  SubRange<out_edge_iterator> out_edges(Vertex* u) const;

  /// Returns the edges incoming to a vertex.
  SubRange<in_edge_iterator> in_edges(Vertex* u) const;

  /// Returns the vertices adjacent to u.
  SubRange<adjacency_iterator> adjacent_vertices(Vertex* u) const;

  /// Returns the range of all vertices.
  SubRange<vertex_iterator> vertices() const;

  /// Returns the range of all vertices.
  SubRange<edge_iterator> edges() const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the number of vertices
  size_t num_vertices() const;

  /// Returns the number of edges
  size_t num_edges() const;

  /// Returns the number of edges adjacent to a vertex.
  size_t in_degree(Vertex* u) const;

  /// Returns the number of edges adjacent to a vertex.
  size_t out_degree(Vertex* u) const;

  /// Returns the number of edges adjacent to a vertex.
  size_t degree(Vertex* u) const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Vertex* u) const;

  /// Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Vertex* u, Vertex* v) const;

  /// Returns true if the graph contains an undirected edge.
  bool contains(edge_descriptor e) const;

  /// Returns the first vertex or the null vertex if the graph is empty.
  Vertex* root() const;

  /// Returns an opaque reference to the property associated with a vertex.
  OpaqueRef property(Vertex* u);

  /// Returns an opaque const reference to the property associated with a vertex.
  OpaqueCref property(Vertex* u) const;

  /// Returns an opaque reference to the property associated with an edge.
  OpaqueRef property(edge_descriptor e);

  /// Returns an opaque const reference to the property associated with an edge.
  OpaqueCref property(edge_descriptor e) const;

  /// Returns true if two cluster graphs are identical.
  bool operator==(const ClusterGraph& other) const;

  /// Returns true if two cluster graphs are not identical.
  bool operator!=(const ClusterGraph& other) const;

  // Domain accessors
  //--------------------------------------------------------------------------

  /// Returns the cardinality of the union of all the clusters.
  size_t num_arguments() const;

  /// Returns the number of clusters in which the argument occurs.
  size_t count(Arg x) const;

  /// Returns the union of all the clusters in this graph.
  SubRange<argument_iterator> arguments() const;

  /// The cluster associated with a vertex.
  const Domain& cluster(Vertex* v) const;

  /// The seprator associated with an edge.
  const Domain& separator(edge_descriptor e) const;

  /// Returns the shape of the arguments at a vertex.
  Shape shape(Vertex* v, const ShapeMap& map) const;

  /// Returns the shape of the arguments at an edge.
  Shape shape(edge_descriptor e, const ShapeMap& map) const;

  /// Returns the index mapping from a domain to the cluster at a vertex.
  Dims dims(Vertex* v, const Domain& dom) const;

  /// Returns the index mapping from a domain to the separator at an edge.
  Dims dims(edge_descriptor e, const Domain& dom) const;

  /// Returns the index mapping from the separator to the source cluster.
  Dims source_dims(edge_descriptor e) const;

  /// Returns the index mapping from the separator to the target cluster.
  Dims target_dims(edge_descriptor e) const;

  // Queries
  //--------------------------------------------------------------------------

  /**
   * Computes the Markov graph capturing the dependencies in this model.
   */
  MarkovNetwork markov_network() const;

  /**
   * Returns true if the graph is connected.
   */
  bool is_connected();

  /**
   * Returns true if the cluster graph is a tree.
   */
  bool is_tree();

  /**
   * Returns true if the cluster graph satisfies the running intersection
   * property. A cluster graph satisfies the running intersection property
   * if, for each value, the clusters and separators containing that value
   * form a subtree.
   */
  bool has_running_intersection();

  /**
   * Returns true if the cluster graph represents a triangulated model,
   * i.e., is a tree and satisfies the running intersection property.
   */
  bool is_triangulated();

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
  Vertex* find_cluster_cover(const Domain& dom) const;

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
  Vertex* find_cluster_meets(const Domain& dom) const;

  /**
   * Returns an edge whose separator meets (intersects) the supplied
   * domain. The returned edge is the one that has the smallest separator
   * that has maximal intersection with the supplied domain.
   */
  edge_descriptor find_separator_meets(const Domain& dom) const;

  /**
   * Visits the vertices whose clusters overlap the supplied domain.
   */
  void intersecting_clusters(const Domain& dom, VertexVisitor visitor) const;

  /**
   * Visits the edges whose separators overlap the supplied domain.
   */
  void intersecting_separators(const Domain& dom, EdgeVisitor visitor) const;

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
  void mark_subtree_cover(const Domain& domain, bool force_continuous);

  /**
   * Performs a pre-order traversal of a tree starting at the given root,
   * invoking a visitor at the same time as each edge is traversed.
   * The order in which the visitor is invoked is consistent
   * with a breadth-first or depth-first traversal from v.
   *
   * \param start
   *        A vertex of the graph. The traversal is started at this vertex.
   * \param visitor
   *        The visitor that is invoked to each edge traversed.
   *        If the graph is disconnected, then this visitor is applied
   *        only to edges directed away from the root vertex.
   */
  void pre_order_traversal(Vertex* start, EdgeVisitor visitor);

  /**
   * Performs a post-order traversal of a tree, starting at the given root.
   * The given edge visitor is applied to each edge during the traversal.
   * The reverse of the order in which the visitor is applied to the edges
   * is consistent with a breadth-first or depth-first traversal from root.
   *
   * \param start
   *        A vertex of the graph. The traversal is started at this vertex.
   * \param edge_visitor
   *        The visitor that is applied to each edge of the graph.
   *        If the graph is disconnected, then this visitor is applied
   *        only to edges directed toward the start vertex.
   */
  void post_order_traversal(Vertex* start, EdgeVisitor visitor);

  /**
   * Visits each (directed) edge of the cluster graph once in a traversal
   * such that each \f$v \rightarrow w\f$ is visited after all edges
   * \f$u \rightarrow v\f$ (with \f$u \neq w\f$) are visited. Orders
   * of this type are said to satisfy the "message passing protocol"
   * (MPP).
   *
   * \param start
   *        A vertex of the graph. The traversal is started at this vertex.
   *        Can be a null vertex to denote arbitrary root.
   * \param visitor
   *        The visitor that is applied to each edge of the graph.
   *        If the graph is disconnected, then this visitor is applied
   *        only to edges directed towards / from the start vertex.
   */
  void mpp_traversal(Vertex* root, EdgeVisitor visitor);

  // Mutating operations
  //--------------------------------------------------------------------------

  /**
   * Adds a vertex with the given cluster.
   */
  Vertex* add_vertex(Domain cluster);

  /**
   * Adds an edge {u, v} to the graph with the given separator. The edge
   * endpoints must exist, and the separator must be a subset of the clusters
   * at these vertices. If the edge already exists, does not perform anything.
   * \return the edge and bool indicating whether the insertion took place
   */
  edge_descriptor add_edge(Vertex* u, Vertex* v, Domain separator);

  /**
   * Adds an edge {u, v} to the graph, setting the separator to the
   * intersection of the two clusters at the endpoints.
   * If the edge already exists, does not perform anything.
   * \return the edge and bool indicating whether the insertion took place
   */
  edge_descriptor add_edge(Vertex* u, Vertex* v);

  /**
   * Updates the cluster associated with an existing vertex.
   */
  void update_cluster(Vertex* u, const Domain& cluster);

  /**
   * Updates the separator associated with an edge.
   */
  void update_separator(edge_descriptor e, const Domain& separator);

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
  Vertex* merge(edge_descriptor e);

  /// Removes a vertex and the associated cluster and property.
  void remove_vertex(Vertex* v);

  /// Removes an undirected edge {u, v} and the associated data.
  void remove_edge(Vertex* u, Vertex* v);

  /// Removes an undirected edge and the associated data.
  void remove_edge(edge_descriptor e);

  /// Removes all edges incindent to a vertex
  void clear_vertex(Vertex* u);

  /// Removes all edges from the graph
  void remove_edges();

  /// Removes all vertices and edges from the graph
  void clear();

  /// Resets the color of all vertices to white.
  void reset_color();

  // Triangulation
  //--------------------------------------------------------------------------

  /**
   * Initializes this cluster graph to a junction tree that represents the
   * triangulation of the supplied graph.
   *
   * \tparam Graph an undirected graph type whose vertices correspond to
   *               variables stored in Domain.
   * \tparam Strategy a type that models the EliminationStrategy concept
   */
  void triangulated(MarkovNetwork& mn, const EliminationStrategy& strategy);

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

private:
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }

}; // class ClusterGraph

/// Returns the color associated with a vertex.
boost::default_color_type get(const ClusterGraph::VertexColorMap&, ClusterGraph::Vertex* v);

/// Sets the color associated with a vertex.
void put(const ClusterGraph::VertexColorMap&, ClusterGraph::Vertex* v, boost::default_color_type c);

/// Returns the index associated with a vertex.
size_t get(const ClusterGraph::VertexIndexMap&, ClusterGraph::Vertex* v);

/**
 * A cluster graph with strongly typed vertex and edge properties.
 *
 * \tparam VP
 *         A type of values stored at each vertex.
 * \tparam EP
 *         A type of values stored at each edge.
 */
template <typename VP, typename EP = VP>
struct ClusterGraphT : ClusterGraph {
  static_assert(!std::is_void_v<VP>, "VP must be a non-void property type.");
  static_assert(!std::is_void_v<EP>, "EP must be a non-void property type.");

  using ClusterGraph::add_vertex;
  using ClusterGraph::add_edge;

  ClusterGraphT()
    : ClusterGraph(property_layout<VP>(), property_layout<EP>()) {}

  VP& operator[](Vertex* u) {
    return opaque_cast<VP>(property(u));
  }

  const VP& operator[](Vertex* u) const {
    return opaque_cast<VP>(property(u));
  }

  EP& operator[](edge_descriptor e) {
    return opaque_cast<EP>(property(e));
  }

  const EP& operator[](edge_descriptor e) const {
    return opaque_cast<EP>(property(e));
  }

  Vertex* add_vertex(Domain cluster, VP vp) {
    Vertex* v = ClusterGraph::add_vertex(std::move(cluster));
    (*this)[v] = std::move(vp);
    return v;
  }

  edge_descriptor add_edge(Vertex* u, Vertex* v, Domain separator, EP ep) {
    edge_descriptor e = ClusterGraph::add_edge(u, v, std::move(separator));
    (*this)[e] = std::move(ep);
    return e;
  }

  edge_descriptor add_edge(Vertex* u, Vertex* v, EP ep) {
    edge_descriptor e = ClusterGraph::add_edge(u, v);
    (*this)[e] = std::move(ep);
    return e;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const ClusterGraph>(this));

    ar(cereal::make_size_tag(num_vertices()));
    for (Vertex* v : vertices()) {
      ar(operator[](v));
    }

    ar(cereal::make_size_tag(num_edges()));
    for (edge_descriptor e : edges()) {
      ar(operator[](e));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<ClusterGraph>(this));

    cereal::size_type vertex_count;
    ar(cereal::make_size_tag(vertex_count));
    assert(vertex_count == num_vertices());
    for (Vertex* v : vertices()) {
      ar(operator[](v));
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    assert(edge_count == num_edges());
    for (edge_descriptor e : edges()) {
      ar(operator[](e));
    }
  }
};

} // namespace libgm
