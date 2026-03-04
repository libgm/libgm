#pragma once

#include <libgm/object.hpp>
#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/subrange.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/graph/directed_edge.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/bind1_iterator.hpp>
#include <libgm/iterator/bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>

#include <cassert>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

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
class BayesianNetwork : public Object {
private:
  /**
   * A struct with the data associated with each vertex. This structure
   * stores the property associated with the vertex as well all edges from/to
   * parent and child vertices (along with the edge properties).
   */
  struct VertexData;

  /// The map type used to associate neighbors and edge data with each vertex.
  using AdjacencySet = ankerl::unordered_dense::set<Arg>;

  /// The map types that associate all the vertices with their VertexData.
  using VertexDataMap = ankerl::unordered_dense::map<Arg, VertexData*>;

public:
  /// The implementation class.
  struct Impl;

  Impl& impl();
  const Impl& impl() const;

  VertexData& data(Arg arg);
  const VertexData& data(Arg arg) const;

  // Descriptors
  using vertex_descriptor = Arg;
  using edge_descriptor   = DirectedEdge<Arg>;

  // Iterators (the exact types are implementation detail)
  using out_edge_iterator  = Bind1Iterator<AdjacencySet::const_iterator, edge_descriptor>;
  using in_edge_iterator   = Bind2Iterator<Domain::const_iterator, edge_descriptor>;
  using adjacency_iterator = AdjacencySet::const_iterator;
  using vertex_iterator    = MapKeyIterator<VertexDataMap>;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty Bayesian network.
  explicit BayesianNetwork(size_t count = 0);

protected:
  BayesianNetwork(size_t count, PropertyLayout layout);

public:

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the outgoing edges from a vertex.
  SubRange<out_edge_iterator> out_edges(Arg u) const;

  /// Returns the edges incoming to a vertex.
  SubRange<in_edge_iterator> in_edges(Arg u) const;

  /// Returns the children of u.
  SubRange<adjacency_iterator> adjacent_vertices(Arg u) const;

  /// Returns the range of all vertices.
  SubRange<vertex_iterator> vertices() const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Arg u) const;

  /// Returns true if the graph contains a directed edge (u, v).
  bool contains(Arg u, Arg v) const;

  /// Returns true if the graph contains a directed edge.
  bool contains(const DirectedEdge<Arg> e) const;

  /// Returns a directed edge (u,v) between two vertices or null edge if one does not exist.
  DirectedEdge<Arg> edge(Arg u, Arg v) const;

  /// Returns the number of outgoing edges to a vertex.
  size_t out_degree(Arg u) const;

  /// Returns the number of incoming edges to a vertex.
  size_t in_degree(Arg u) const;

  /// Returns the total number of edges adjacent to a vertex.
  size_t degree(Arg u) const;

  /// Returns true if the graph has no vertices.
  bool empty() const;

  /// Returns the number of vertices.
  size_t num_vertices() const;

  /// Returns the number of edges.
  size_t num_edges() const;

  /// Returns the arguments of a factor.
  const Domain& parents(Arg u) const;

  /// Returns the raw pointer to the property associated with a vertex.
  void* property(Arg u);

  /// Returns the raw pointer to the property associated with a vertex.
  const void* property(Arg u) const;

  // Queries
  //--------------------------------------------------------------------------

  /**
   * Computes a minimal Markov graph capturing dependencies in this model.
   */
  MarkovNetwork markov_network() const;

  // Modifications
  //--------------------------------------------------------------------------

  /// Adds an argument with the given parent domain.
  bool add_vertex(Arg u, Domain parents);

public:
  /// Removes a vertex from the graph, provided that it has no outgoing edges.
  void remove_vertex(Arg u);

  /// Removes all edges incoming to a vertex.
  void remove_in_edges(Arg u);

  /// Removes all vertices and edges from the graph.
  void clear();

}; // class BayesianNetwork

/**
 * Bayesian network with strongly-typed vertices.
 */
template <typename VP>
struct BayesianNetworkT : BayesianNetwork {
  static_assert(!std::is_void_v<VP>, "VP must be a non-void property type.");

  using BayesianNetwork::add_vertex;

  explicit BayesianNetworkT(size_t count = 0)
    : BayesianNetwork(count, property_layout<VP>()) {}

  VP& operator[](Arg u) {
    return *static_cast<VP*>(BayesianNetwork::property(u));
  }

  const VP& operator[](Arg u) const {
    return *static_cast<const VP*>(BayesianNetwork::property(u));
  }

  bool add_vertex(Arg u, Domain parents, VP vp) {
    bool inserted = BayesianNetwork::add_vertex(u, std::move(parents));
    (*this)[u] = std::move(vp);
    return inserted;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const BayesianNetwork>(this));
    ar(cereal::make_size_tag(num_vertices()));
    for (Arg u : vertices()) {
      ar(CEREAL_NVP(u), cereal::make_nvp("property", operator[](u)));
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<BayesianNetwork>(this));
    cereal::size_type n;
    ar(cereal::make_size_tag(n));
    for (size_t i = 0; i < n; ++i) {
      Arg u;
      ar(CEREAL_NVP(u));
      assert(contains(u));
      ar(cereal::make_nvp("property", operator[](u)));
    }
  }
}; // struct BayesianNetworkT

} // namespace libgm
