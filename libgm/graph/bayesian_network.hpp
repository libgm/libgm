#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/graph/UndirectedGraph.hpp>
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
class BayesianNetwork : public Object {
private:
  /**
   * A struct with the data associated with each vertex. This structure
   * stores the property associated with the vertex as well all edges from/to
   * parent and child vertices (along with the edge properties).
   */
  struct Vertex;

  /// The map type used to associate neighbors and edge data with each vertex.
  using AdjacencySet = ankerl::unordered_dense::set<Arg>;

  /// The map types that associates all the vertices with their VertexData.
  using VertexDataMap = ankerl::unordered_dense::map<Arg, Vertex*>;

  /// The implementation object.
  struct Impl;

  Impl& impl();
  const Impl& impl() const;

  Vertex& data(Arg arg);
  const Vertex& data(Arg arg) const;

  //--------------------------------------------------------------------------
public:
  // Descriptors
  using vertex_descriptor = Arg;
  using edge_descriptor   = DirectedEdge<Arg>;

  // Iterators (the exact types are implementation detail)
  using out_edge_iterator  = Bind1Iterator<NeighborSet::const_iterator, edge_descriptor>;
  using in_edge_iterator   = Bind2Iterator<Domain::const_iterator, edge_descriptor>;
  using adjacency_iterator = NeighborSet::const_iterator;
  using vertex_iterator    = MapKeyIterator<VertexDataMap>;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty Bayesian network.
  explicit BayesianNetwork(size_t count = 0);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the outgoing edges from a vertex.
  boost::iterator_range<out_edge_iterator> out_edges(Arg u) const;

  /// Returns the edges incoming to a vertex.
  boost::iterator_range<in_edge_iterator> in_edges(Arg u) const;

  /// Returns the children of u.
  boost::iterator_range<adjacency_iterator> adjacent_vertices(Arg u) const;

  /// Returns the range of all vertices.
  boost::iterator_range<vertex_iterator> vertices() const;

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

  /// Returns the property associated with a vertex.
  Object& operator[](Arg u);

  /// Returns the property associated with a vertex.
  const Object& operator[](Arg u) const;

  // Queries
  //--------------------------------------------------------------------------

  /**
   * Computes a minimal Markov graph capturing dependencies in this model.
   */
  MarkovNetworkT<void, void> markov_network() const;

  // Modifications
  //--------------------------------------------------------------------------

  /**
   * Adds an argument to the graph with the given domain and associates a property with
   * that vertex. If the vertex is already present, its property and edges are overwritten.
   *
   * \returns true if the vertex was newly inserted
   */
  bool add_vertex(Arg u, Domain parents, Object object = Object());

  /// Removes a vertex from the graph, provided that it has no outgoing edges.
  void remove_vertex(Arg u);

  /// Removes all edges incoming to a vertex.
  void remove_in_edges(Arg u);

  /// Removes all vertices and edges from the graph.
  void clear();

}; // class BayesianNetwork

} // namespace libgm
