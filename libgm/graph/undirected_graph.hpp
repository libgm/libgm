#pragma once

#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/graph/intrusive_edge.hpp>
#include <libgm/graph/util/bgl.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/casting_iterator.hpp>
#include <libgm/iterator/member_iterator.hpp>
#include <libgm/opaque.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>

#include <functional>
#include <memory>
#include <ranges>

namespace libgm {

/**
 * A generic undirected graph with intrusive vertex and edge storage.
 *
 * Vertices and edges can carry optional trailing properties controlled by
 * `PropertyLayout`, but the graph itself is not templated.
 */
class UndirectedGraph {
public:
  /// The implementation type.
  struct Impl;

  /// The graph vertex type.
  struct Vertex;

  /// The graph edge type.
  struct Edge;

  /// Outputs a vertex descriptor to a stream.
  friend std::ostream& operator<<(std::ostream& out, Vertex* v);

  /// Outputs an edge descriptor to a stream.
  friend std::ostream& operator<<(std::ostream& out, Edge* e);

  // Descriptors
  //--------------------------------------------------------------------------
  /// Handle to a vertex.
  using vertex_descriptor = Vertex*;

  /// Directed view of an undirected edge.
  using edge_descriptor = IntrusiveEdge<Vertex, Edge>;

  // Graph categories
  //--------------------------------------------------------------------------
  using directed_category = boost::directed_tag;
  using edge_parallel_category = boost::disallow_parallel_edge_tag;
  struct traversal_category :
    public virtual boost::vertex_list_graph_tag,
    public virtual boost::incidence_graph_tag,
    public virtual boost::adjacency_graph_tag,
    public virtual boost::edge_list_graph_tag { };

  // Size types
  //--------------------------------------------------------------------------
  using vertices_size_type = size_t;
  using edges_size_type = size_t;
  using degree_size_type = size_t;

  // Iterators
  //--------------------------------------------------------------------------
  /// Iterator over outgoing directed views of an incident edge.
  using out_edge_iterator = CastingIterator<IntrusiveList<Edge>::entry_iterator, edge_descriptor>;

  /// Iterator over incoming directed views of an incident edge.
  using in_edge_iterator = MemberIterator<out_edge_iterator, &edge_descriptor::reverse>;

  /// Iterator over adjacent vertices.
  using adjacency_iterator = MemberIterator<out_edge_iterator, &edge_descriptor::target>;

  /// Iterator over all vertices.
  using vertex_iterator = IntrusiveList<Vertex>::iterator;

  /// Iterator over all edges.
  using edge_iterator = IntrusiveList<Edge>::iterator;

  // Color map for vertices
  //--------------------------------------------------------------------------
  struct VertexColorMap {
    using value_type = boost::default_color_type;
    using reference = boost::default_color_type;
    using key_type = Vertex*;
    using category = boost::read_write_property_map_tag;
  };

  // Index map for vertices
  //--------------------------------------------------------------------------
  struct VertexIndexMap {
    using value_type = size_t;
    using reference = size_t;
    using key_type = Vertex*;
    using category = boost::readable_property_map_tag;
  };

  /// Visitor invoked during edge traversals.
  using EdgeVisitor = std::function<void(edge_descriptor)>;

  // Constructors and destructors
  //--------------------------------------------------------------------------
  UndirectedGraph();
  UndirectedGraph(const UndirectedGraph& other);
  UndirectedGraph(UndirectedGraph&& other) noexcept;
  UndirectedGraph& operator=(const UndirectedGraph& other);
  UndirectedGraph& operator=(UndirectedGraph&& other) noexcept;
  ~UndirectedGraph();

protected:
  UndirectedGraph(PropertyLayout vertex_layout, PropertyLayout edge_layout);

public:
  /// Returns the null vertex.
  static Vertex* null_vertex() { return nullptr; }

  /// Returns the color property map used by BGL traversals.
  VertexColorMap vertex_color_map() { return {}; }

  /// Assigns consecutive indices to vertices and returns the index property map.
  VertexIndexMap vertex_index_map();

  // Accessors
  //--------------------------------------------------------------------------
  /// Returns the outgoing edge views adjacent to `u`.
  std::ranges::subrange<out_edge_iterator> out_edges(Vertex* u) const;

  /// Returns the incoming edge views adjacent to `u`.
  std::ranges::subrange<in_edge_iterator> in_edges(Vertex* u) const;

  /// Returns the vertices adjacent to `u`.
  std::ranges::subrange<adjacency_iterator> adjacent_vertices(Vertex* u) const;

  /// Returns the range of all vertices.
  std::ranges::subrange<vertex_iterator> vertices() const;

  /// Returns the range of all edges.
  std::ranges::subrange<edge_iterator> edges() const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns true if the graph contains the supplied vertex.
  bool contains(Vertex* u) const;

  /// Returns true if the graph contains an edge between `u` and `v`.
  bool contains(Vertex* u, Vertex* v) const;

  /// Returns true if the graph contains the supplied edge.
  bool contains(edge_descriptor e) const;

  /// Returns the number of vertices.
  size_t num_vertices() const;

  /// Returns the number of edges.
  size_t num_edges() const;

  /// Returns the incoming degree of `u`.
  size_t in_degree(Vertex* u) const;

  /// Returns the outgoing degree of `u`.
  size_t out_degree(Vertex* u) const;

  /// Returns the degree of `u`.
  size_t degree(Vertex* u) const;

  /// Returns the first vertex or the null vertex if the graph is empty.
  Vertex* root() const;

  /// Returns the current mark bit for a vertex.
  bool marked(Vertex* v) const;

  /// Returns the current mark bit for an edge.
  bool marked(edge_descriptor e) const;

  /// Returns an opaque reference to the property associated with a vertex.
  OpaqueRef property(Vertex* u);

  /// Returns an opaque const reference to the property associated with a vertex.
  OpaqueCref property(Vertex* u) const;

  /// Returns an opaque reference to the property associated with an edge.
  OpaqueRef property(edge_descriptor e);

  /// Returns an opaque const reference to the property associated with an edge.
  OpaqueCref property(edge_descriptor e) const;

  /// Returns true if the graph is connected.
  bool is_connected();

  /// Returns true if the graph is a tree.
  bool is_tree();

  /// Performs a pre-order traversal from `start`.
  void pre_order_traversal(Vertex* start, EdgeVisitor visitor);

  /// Performs a post-order traversal from `start`.
  void post_order_traversal(Vertex* start, EdgeVisitor visitor);

  /// Performs a message-passing-protocol traversal from `start`.
  void mpp_traversal(Vertex* start, EdgeVisitor visitor);

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds a vertex to the graph.
  Vertex* add_vertex();

  /// Adds an edge between `u` and `v`.
  edge_descriptor add_edge(Vertex* u, Vertex* v);

  /// Removes a vertex and all of its incident edges.
  void remove_vertex(Vertex* u);

  /// Removes an edge descriptor if present. Returns 1 if removed, 0 otherwise.
  size_t remove_edge(edge_descriptor e);

  /// Removes the edge between `u` and `v` if present. Returns 1 if removed, 0 otherwise.
  size_t remove_edge(Vertex* u, Vertex* v);

  /// Removes all edges incident to `u`.
  void clear_vertex(Vertex* u);

  /// Removes all edges from the graph.
  void remove_edges();

  /// Removes all vertices and edges from the graph.
  void clear();

  /// Resets every vertex color to white.
  void reset_color();

protected:
  Impl& impl();
  const Impl& impl() const;
  Vertex& data(Vertex* u);
  const Vertex& data(Vertex* u) const;
  Edge& data(edge_descriptor e);
  const Edge& data(edge_descriptor e) const;
  void set_marked(Vertex* v, bool value);
  void set_marked(edge_descriptor e, bool value);

private:
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }
};

/// Prints the graph to an output stream.
std::ostream& operator<<(std::ostream& out, const UndirectedGraph& graph);

/// Returns the color associated with a vertex.
boost::default_color_type get(const UndirectedGraph::VertexColorMap&, UndirectedGraph::Vertex* v);

/// Sets the color associated with a vertex.
void put(const UndirectedGraph::VertexColorMap&, UndirectedGraph::Vertex* v, boost::default_color_type c);

/// Returns the index associated with a vertex.
size_t get(const UndirectedGraph::VertexIndexMap&, UndirectedGraph::Vertex* v);

} // namespace libgm
