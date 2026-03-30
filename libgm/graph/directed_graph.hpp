#pragma once

#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/graph/directed_edge.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/bind1_iterator.hpp>
#include <libgm/iterator/bind2_iterator.hpp>
#include <libgm/opaque.hpp>

#include <ankerl/unordered_dense.h>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <boost/graph/graph_traits.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace libgm {

/**
 * A directed graph with fixed-size parent lists and intrusive child lists.
 *
 * Each vertex stores its incoming neighbors as a `std::vector<Vertex*>`.
 * Each source vertex stores its outgoing neighbors as an `IntrusiveList<Vertex>`,
 * where the hook for edge `(u, v)` is stored inside `v`. This makes the
 * incoming adjacency compact and keeps child traversal efficient.
 *
 * Vertices can carry an optional trailing property controlled by
 * `PropertyLayout`, but the graph itself is not templated.
 */
class DirectedGraph {
public:
  /// The vertex type (`Vertex*` is the public descriptor).
  struct Vertex;

  /// The implementation class.
  struct Impl;

  // Descriptors
  //--------------------------------------------------------------------------
  using vertex_descriptor = Vertex*;
  using edge_descriptor = DirectedEdge<Vertex*>;

  // Iterators (the exact types are implementation detail)
  //--------------------------------------------------------------------------
  using out_edge_iterator =
    Bind1Iterator<IntrusiveList<Vertex>::iterator, edge_descriptor, Vertex*>;
  using in_edge_iterator =
    Bind2Iterator<std::vector<Vertex*>::const_iterator, edge_descriptor, Vertex*>;
  using adjacency_iterator = IntrusiveList<Vertex>::iterator;
  using vertex_iterator = IntrusiveList<Vertex>::iterator;

  // Graph categories
  //--------------------------------------------------------------------------
  using directed_category = boost::directed_tag;
  using edge_parallel_category = boost::disallow_parallel_edge_tag;
  struct traversal_category :
    public virtual boost::vertex_list_graph_tag,
    public virtual boost::incidence_graph_tag,
    public virtual boost::adjacency_graph_tag { };

  // Size types
  //--------------------------------------------------------------------------
  using vertices_size_type = size_t;
  using edges_size_type = size_t;
  using degree_size_type = size_t;

  /// Returns the null vertex.
  static Vertex* null_vertex() { return nullptr; }

  // Constructors and destructors
  //--------------------------------------------------------------------------
  DirectedGraph();
  DirectedGraph(const DirectedGraph& other);
  DirectedGraph(DirectedGraph&& other) noexcept;
  DirectedGraph& operator=(const DirectedGraph& other);
  DirectedGraph& operator=(DirectedGraph&& other) noexcept;
  ~DirectedGraph();

protected:
  explicit DirectedGraph(PropertyLayout layout);

public:
  // Accessors
  //--------------------------------------------------------------------------
  /// Returns the outgoing edges from a vertex.
  std::ranges::subrange<out_edge_iterator> out_edges(Vertex* u) const;

  /// Returns the incoming edges to a vertex.
  std::ranges::subrange<in_edge_iterator> in_edges(Vertex* u) const;

  /// Returns the children of a vertex.
  std::ranges::subrange<adjacency_iterator> adjacent_vertices(Vertex* u) const;

  /// Returns the range of all vertices.
  std::ranges::subrange<vertex_iterator> vertices() const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Vertex* u) const;

  /// Returns true if the graph contains the directed edge `(u, v)`.
  bool contains(Vertex* u, Vertex* v) const;

  /// Returns true if the graph contains the given edge.
  bool contains(edge_descriptor e) const;

  /// Returns the edge `(u, v)` or the null edge if it does not exist.
  edge_descriptor edge(Vertex* u, Vertex* v) const;

  /// Returns the number of incoming edges to a vertex.
  size_t in_degree(Vertex* u) const;

  /// Returns the number of outgoing edges from a vertex.
  size_t out_degree(Vertex* u) const;

  /// Returns the total degree of a vertex.
  size_t degree(Vertex* u) const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the number of vertices.
  size_t num_vertices() const;

  /// Returns the number of edges.
  size_t num_edges() const;

  /// Returns the parents of a vertex.
  const std::vector<Vertex*>& parents(Vertex* u) const;

  /// Returns an opaque reference to the vertex property.
  OpaqueRef property(Vertex* u);

  /// Returns an opaque const reference to the vertex property.
  OpaqueCref property(Vertex* u) const;

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds a new vertex with the specified parents.
  Vertex* add_vertex(std::vector<Vertex*> parents = {});

  /// Replaces the parent list of an existing vertex.
  void set_parents(Vertex* u, std::vector<Vertex*> parents);

  /// Removes a vertex if it has no outgoing edges.
  size_t remove_vertex(Vertex* u);

  /// Removes all incoming edges to a vertex.
  void remove_in_edges(Vertex* u);

  /// Removes all vertices and edges from the graph.
  void clear();

protected:
  Impl& impl();
  const Impl& impl() const;
  Vertex& data(Vertex* u);
  const Vertex& data(Vertex* u) const;

private:
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }
};

} // namespace libgm
