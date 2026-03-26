#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_bind1_iterator.hpp>
#include <libgm/iterator/map_bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/opaque.hpp>

#include <ankerl/unordered_dense.h>

#include <boost/graph/graph_traits.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <iosfwd>
#include <memory>
#include <new>
#include <ranges>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

/**
 * A class that representeds an undirected graph as an adjancy list (map).
 * The template is paraemterized by the vertex type as well as the type
 * of properties associated with vertices and edges.
 *
 * \ingroup graph_types
 */
class MarkovNetwork {
private:
  /// The data associated with a vertex.
  struct VertexData;

  /// The implementation class.
  struct Impl;

  Impl& impl();
  const Impl& impl() const;

  VertexData& data(Arg arg);
  const VertexData& data(Arg arg) const;

  /// The map type used to associate neighbors and edge data with each vertex.
  using AdjacencyMap = ankerl::unordered_dense::map<Arg, void*>;

  /// The map types that associates all the vertices with their VertexData.
  using VertexDataMap = ankerl::unordered_dense::map<Arg, VertexData*>;

  // Graph concept typedefs
  //--------------------------------------------------------------------------
public:
  // Descriptors
  using vertex_descriptor = Arg;
  using edge_descriptor   = UndirectedEdge<Arg>;

  // Iterators (the exact types are implementation detail)
  using out_edge_iterator  = MapBind1Iterator<AdjacencyMap, edge_descriptor>;
  using in_edge_iterator   = MapBind2Iterator<AdjacencyMap, edge_descriptor>;
  using adjacency_iterator = MapKeyIterator<AdjacencyMap>;
  using vertex_iterator    = MapKeyIterator<VertexDataMap>;

  // Graph categories
  using directed_category = boost::undirected_tag;
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

  /// Visitor
  using VertexVisitor = std::function<void(Arg)>;

  // Constructors and destructors
  //--------------------------------------------------------------------------
public:
  /// Create an empty graph.
  explicit MarkovNetwork(size_t count = 0);

  MarkovNetwork(const MarkovNetwork& g);
  MarkovNetwork(MarkovNetwork&& g) noexcept;
  MarkovNetwork& operator=(const MarkovNetwork& g);
  MarkovNetwork& operator=(MarkovNetwork&& g) noexcept;
  ~MarkovNetwork();

protected:
  MarkovNetwork(size_t count, PropertyLayout vertex_layout, PropertyLayout edge_layout);

public:
  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the null vertex.
  static Arg null_vertex() { return Arg(); }

  /// Returns the edges outgoing from a vertex.
  std::ranges::subrange<out_edge_iterator> out_edges(Arg u) const;

  /// Returns the edges incoming to a vertex.
  std::ranges::subrange<in_edge_iterator> in_edges(Arg u) const;

  /// Returns the vertices adjacent to u.
  std::ranges::subrange<adjacency_iterator> adjacent_vertices(Arg u) const;

  /// Returns the range of all vertices.
  std::ranges::subrange<vertex_iterator> vertices() const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Arg u) const;

  /// Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Arg u, Arg v) const;

  /// Returns true if the graph contains an undirected edge.
  bool contains(const UndirectedEdge<Arg>& e) const;

  /// Returns an undirected edge (u, v). The edge must exist.
  UndirectedEdge<Arg> edge(Arg u,  Arg v) const;

  /// Returns the number of edges adjacent to a vertex.
  size_t out_degree(Arg u) const;

  /// Returns the number of edges adjacent to a vertex.
  size_t in_degree(Arg u) const;

  /// Returns the number of edges adjacent to a vertex.
  size_t degree(Arg u) const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the number of vertices.
  size_t num_vertices() const;

  /// Returns the number of edges.
  size_t num_edges() const;

  /// Returns an opaque reference to the property associated with a vertex.
  OpaqueRef property(Arg u);

  /// Returns an opaque const reference to the property associated with a vertex.
  OpaqueCref property(Arg u) const;

  /// Returns an opaque reference to the property associated with an edge.
  OpaqueRef property(const UndirectedEdge<Arg>& e);

  /// Returns an opaque const reference to the property associated with an edge.
  OpaqueCref property(const UndirectedEdge<Arg>& e) const;

  /// Returns a Markov network with the same structure but omitting the properties.
  MarkovNetwork without_properties() const;

  // Modifications
  //--------------------------------------------------------------------------

  /**
   * Adds a vertex to a graph.
   * \returns true if the insertion took place (i.e., vertex was not present).
   */
  bool add_vertex(Arg u);

  /**
   * Adds an edge {u,v} to the graph. If the edge already exists, the existing
   * edge is returned. If u and v are not present, they are added.
   * \return the edge and bool indicating whether the edge was newly added.
   */
  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v);

  /// Adds edges from given source vertex to all specified target vertices.
  void add_edges(Arg u, const std::vector<Arg>& vs);

  /// Adds edges among all given vertices.
  void add_clique(const Domain& vertices);

  /// Removes a vertex from the graph and all its incident edges.
  /// Returns 1 if removed, 0 if the vertex does not exist.
  size_t remove_vertex(Arg u);

  /// Removes an undirected edge {u, v}.
  /// Returns 1 if removed, 0 if the edge does not exist.
  size_t remove_edge(Arg u, Arg v);

  /// Removes all edges incident to a vertex.
  void remove_edges(Arg u);

  /// Removes all edges from the graph.
  void remove_edges();

  /// Removes all vertices and edges from the graph.
  void clear();

  /**
   * Runs the vertex elimination algorithm on a graph. The algorithm eliminates
   * each node from the graph; eliminating a node involves connecting the node's
   * neighbors into a new clique and then removing the node from the graph.
   * The nodes are eliminated greedily in the order specified by the elimination
   * strategy.
   */
  void eliminate(const EliminationStrategy& strategy, VertexVisitor visitor);

  /// Prints a Markov network to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const MarkovNetwork& g);

private:
  std::unique_ptr<Impl> impl_;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(impl_);
  }

}; // class MarkovNetwork

/**
 * A class that represents a Markov network with strongly typed vertex and edge properties.
 *
 * \tparam VP
 *         The type of data associated with vertices.
 * \tparam EP
 *         The type of data associated with edges.
 *
 * \ingroup graph_types
 */
template <typename VP, typename EP = VP>
struct MarkovNetworkT : MarkovNetwork {
  static_assert(!std::is_void_v<VP>, "VP must be a non-void property type.");
  static_assert(!std::is_void_v<EP>, "EP must be a non-void property type.");

  using MarkovNetwork::add_vertex;
  using MarkovNetwork::add_edge;

  MarkovNetworkT(size_t count = 0)
    : MarkovNetwork(count, property_layout<VP>(), property_layout<EP>()) {}

  VP& operator[](Arg u) {
    return opaque_cast<VP>(property(u));
  }

  const VP& operator[](Arg u) const {
    return opaque_cast<VP>(property(u));
  }

  EP& operator[](const UndirectedEdge<Arg>& e) {
    return opaque_cast<EP>(property(e));
  }

  const EP& operator[](const UndirectedEdge<Arg>& e) const {
    return opaque_cast<EP>(property(e));
  }

  bool add_vertex(Arg u, VP vp) {
    bool inserted = MarkovNetwork::add_vertex(u);
    if (inserted) {
      (*this)[u] = std::move(vp);
    }
    return inserted;
  }

  std::pair<UndirectedEdge<Arg>, bool> add_edge(Arg u, Arg v, EP ep) {
    auto result = MarkovNetwork::add_edge(u, v);
    if (result.second) {
      (*this)[result.first] = std::move(ep);
    }
    return result;
  }

  void init_vertices(const std::function<VP(Arg)>& init_fn) {
    for (Arg u : vertices()) {
      (*this)[u] = init_fn(u);
    }
  }

  void init_edges(const std::function<EP(UndirectedEdge<Arg>)>& init_fn) {
    for (Arg u : vertices()) {
      for (UndirectedEdge<Arg> e : out_edges(u)) {
        if (e.is_nominal()) {
          (*this)[e] = init_fn(e);
        }
      }
    }
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const MarkovNetwork>(this));

    ar(cereal::make_size_tag(num_vertices()));
    for (Arg u : vertices()) {
      ar(CEREAL_NVP(u), cereal::make_nvp("property", operator[](u)));
    }

    ar(cereal::make_size_tag(num_edges()));
    for (Arg u : vertices()) {
      for (Arg v : adjacent_vertices(u)) {
        if (u <= v) {
          ar(CEREAL_NVP(u), CEREAL_NVP(v),
             cereal::make_nvp("property", operator[](edge(u, v))));
        }
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<MarkovNetwork>(this));

    cereal::size_type vertex_count;
    ar(cereal::make_size_tag(vertex_count));
    assert(vertex_count == num_vertices());
    for (size_t i = 0; i < vertex_count; ++i) {
      Arg u;
      ar(CEREAL_NVP(u));
      assert(contains(u));
      ar(cereal::make_nvp("property", operator[](u)));
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    assert(edge_count == num_edges());
    for (size_t i = 0; i < edge_count; ++i) {
      Arg u, v;
      ar(CEREAL_NVP(u), CEREAL_NVP(v));
      assert(contains(u, v));
      ar(cereal::make_nvp("property", operator[](edge(u, v))));
    }
  }
};

} // namespace libgm
