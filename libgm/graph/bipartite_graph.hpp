#pragma once

#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/graph/bipartite_edge.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/bind1_iterator.hpp>
#include <libgm/iterator/bind2_iterator.hpp>
#include <libgm/opaque.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <ankerl/unordered_dense.h>

#include <cassert>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace libgm {

/**
 * A generic undirected bipartite graph.
 *
 * The graph stores two kinds of vertices, `Vertex1` and `Vertex2`. Vertex1
 * stores its incident Vertex2 neighbors in an intrusive list. Vertex2 stores
 * its incident Vertex1 neighbors in a fixed `std::vector<Vertex1*>` that is
 * initialized when the vertex is added and never changes afterwards.
 */
class BipartiteGraph {
public:
  /// The vertex types (`Vertex1*` and `Vertex2*` are the public descriptors).
  struct Vertex1;
  struct Vertex2;

  /// The implementation class.
  struct Impl;

  // Descriptors
  //--------------------------------------------------------------------------
  /// Handle to a vertex on the left side of the bipartite graph.
  using vertex1_descriptor = Vertex1*;

  /// Handle to a vertex on the right side of the bipartite graph.
  using vertex2_descriptor = Vertex2*;

  /// Directed view of an edge from `Vertex1` to `Vertex2`.
  using edge12_descriptor = BipartiteEdge<Vertex1*, Vertex2*>;

  /// Directed view of an edge from `Vertex2` to `Vertex1`.
  using edge21_descriptor = BipartiteEdge<Vertex2*, Vertex1*>;

  // Iterators
  //--------------------------------------------------------------------------
  /// Iterator over all left-side vertices.
  using vertex1_iterator = IntrusiveList<Vertex1>::iterator;

  /// Iterator over all right-side vertices.
  using vertex2_iterator = IntrusiveList<Vertex2>::iterator;

  /// Iterator over edges leaving a `Vertex1`.
  using out_edge1_iterator = Bind1Iterator<IntrusiveList<Vertex2>::iterator, edge12_descriptor, Vertex1*>;

  /// Iterator over edges leaving a `Vertex2`.
  using out_edge2_iterator = Bind1Iterator<std::vector<Vertex1*>::const_iterator, edge21_descriptor, Vertex2*>;

  /// Iterator over edges entering a `Vertex1`.
  using in_edge1_iterator = Bind2Iterator<IntrusiveList<Vertex2>::iterator, edge21_descriptor, Vertex1*>;

  /// Iterator over edges entering a `Vertex2`.
  using in_edge2_iterator = Bind2Iterator<std::vector<Vertex1*>::const_iterator, edge12_descriptor, Vertex2*>;

  // Constructors and destructors
  //--------------------------------------------------------------------------
  BipartiteGraph();
  BipartiteGraph(const BipartiteGraph& other);
  BipartiteGraph(BipartiteGraph&& other) noexcept;
  BipartiteGraph& operator=(const BipartiteGraph& other);
  BipartiteGraph& operator=(BipartiteGraph&& other) noexcept;
  ~BipartiteGraph();

protected:
  BipartiteGraph(PropertyLayout vertex1_layout, PropertyLayout vertex2_layout);

public:
  // Accessors
  //--------------------------------------------------------------------------
  /// Returns the range of all left-side vertices.
  std::ranges::subrange<vertex1_iterator> vertices1() const;

  /// Returns the range of all right-side vertices.
  std::ranges::subrange<vertex2_iterator> vertices2() const;

  /// Returns the `Vertex2` neighbors adjacent to `u`.
  const IntrusiveList<Vertex2>& neighbors(Vertex1* u) const;

  /// Returns the `Vertex1` neighbors adjacent to `u`.
  const std::vector<Vertex1*>& neighbors(Vertex2* u) const;

  /// Returns the outgoing edges from a `Vertex1`.
  std::ranges::subrange<out_edge1_iterator> out_edges(Vertex1* u) const;

  /// Returns the incoming edges to a `Vertex1`.
  std::ranges::subrange<in_edge1_iterator> in_edges(Vertex1* u) const;

  /// Returns the outgoing edges from a `Vertex2`.
  std::ranges::subrange<out_edge2_iterator> out_edges(Vertex2* u) const;

  /// Returns the incoming edges to a `Vertex2`.
  std::ranges::subrange<in_edge2_iterator> in_edges(Vertex2* u) const;

  /// Returns true if the graph contains the given `Vertex1`.
  bool contains(Vertex1* u) const;

  /// Returns true if the graph contains the given `Vertex2`.
  bool contains(Vertex2* u) const;

  /// Returns true if the graph contains the edge `{u, v}`.
  bool contains(Vertex1* u, Vertex2* v) const;

  /// Returns the number of adjacent `Vertex2` neighbors of `u`.
  size_t degree(Vertex1* u) const;

  /// Returns the number of adjacent `Vertex1` neighbors of `u`.
  size_t degree(Vertex2* u) const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the number of vertices on the left side.
  size_t num_vertices1() const;

  /// Returns the number of vertices on the right side.
  size_t num_vertices2() const;

  /// Returns an opaque reference to the property associated with `u`.
  OpaqueRef property(Vertex1* u);

  /// Returns an opaque const reference to the property associated with `u`.
  OpaqueCref property(Vertex1* u) const;

  /// Returns an opaque reference to the property associated with `u`.
  OpaqueRef property(Vertex2* u);

  /// Returns an opaque const reference to the property associated with `u`.
  OpaqueCref property(Vertex2* u) const;

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds a new vertex on the left side of the graph.
  Vertex1* add_vertex1();

  /// Adds a new vertex on the right side with the specified neighbors.
  Vertex2* add_vertex2(std::vector<Vertex1*> neighbors);

  /// Removes a `Vertex1` and all incident `Vertex2` vertices.
  void remove_vertex1(Vertex1* u);

  /// Removes a `Vertex2` and all of its incident edges.
  void remove_vertex2(Vertex2* u);

  /// Removes all vertices and edges from the graph.
  void clear();

protected:
  Impl& impl();
  const Impl& impl() const;

private:
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }
};

} // namespace libgm
