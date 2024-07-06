#pragma once

#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/util/vertex_edge_property_iterator.hpp>
#include <libgm/graph/util/void.hpp>
#include <libgm/iterator/map_bind1_iterator.hpp>
#include <libgm/iterator/map_bind2_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/boost::iterator_range.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <boost/graph/graph_traits.hpp>

#include <iterator>
#include <iosfwd>
#include <unordered_map>

namespace libgm {

/**
 * A class that representeds an undirected graph as an adjancy list (map).
 * The template is paraemterized by the vertex type as well as the type
 * of properties associated with vertices and edges.
 *
 * \ingroup graph_types
 */
class MarkovNetwork : public Object {
private:
  struct Vertex;

  /// The map type used to associate neighbors and edge data with each vertex.
  using AdjacencyMap = ankerl::unordered_dense::map<Arg, Object*>;

  /// The map types that associates all the vertices with their VertexData.
  using VertexDataMap = ankerl::unordered_dense::map<Arg, Vertex*>;

  // Graph concept typedefs
  //--------------------------------------------------------------------------
public:
  // Descriptors
  using vertex_descirptor = Arg;
  using edge_descriptor   = UndirectedEdge<Arg>;

  // Iterators (the exact types are implementation detail)
  using out_edge_iterator  = MapBind1Iterator<AdjacencyMap, edge_descriptor>;
  using in_edge_iterator   = MapBind2Iterator<AdjacencyMap, edge_descriptor>;
  using adjacency_iterator = MapKeyIterator<AdjacencyMap>;
  using vertex_iterator    = MapKeyIterator<VertexDataMap>;
  class edge_iterator;

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
  using VertexVisitor = std::function<void(Arg);

  // Constructors and destructors
  //--------------------------------------------------------------------------
public:
  /// Create an empty graph.
  MarkovNetwork();

  /// Copy and move constructors.
  MarkovNetwork(const MarkovNetwork& g) = default;
  MarkovNetwork(MarkovNetwor&& g) = default;

  /// Assignment operators.
  MarkovNetwork& operator=(const MarkovNetwork& g) = default;
  MarkovNetwork& operator=(MarkovNetwork&& g) = default;

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the null vertex.
  static Arg null_vertex() { return Arg(); }

  /// Returns the edges outgoing from a vertex.
  boost::iterator_range<out_edge_iterator> out_edges(Arg u) const;

  /// Returns the edges incoming to a vertex.
  boost::iterator_range<in_edge_iterator> in_edges(Arg u) const;

  /// Returns the vertices adjacent to u.
  boost::iterator_range<adjacency_iterator> adjacent_vertices(Arg u) const;

  /// Returns the range of all vertices.
  boost::iterator_range<vertex_iterator> vertices() const;

  /// Returns the range of all edges in the graph.
  boost::iterator_range<edge_iterator> edges() const;

  /// Returns true if the graph contains the given vertex.
  bool contains(Arg u) const;

  /// Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Arg u, Arg v) const;

  /// Returns true if the graph contains an undirected edge.
  bool contains(const UndirectedEdge<Arg>& e) const;

  /// Returns an undirected edge (u, v). The edge must exist.
  UndirectedEdge<Arg, Object> edge(Arg u,  Arg v) const;

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

  /// Returns the property associated with a vertex.
  Object& operator[](Arg u);

  /// Returns the property associated with a vertex,
  const Object& operator[](Arg u) const;

  /// Returns the property associated with an edge.
  Object& operator[](const UndirectedEdge<Arg>& e);

  /// Returns the property associated with an edge.
  const Object& operator[](const UndirectedEdge<Arg>& e) const;

  // Modifications
  //--------------------------------------------------------------------------

  /**
   * Adds a vertex to a graph and associate the property with that vertex.
   * If the vertex is already present, its property is not overwritten.
   * \returns true if the insertion took place (i.e., vertex was not present).
   */
  bool add_vertex(Arg u, Object object = Object());

  /**
   * Adds an edge {u,v} to the graph. If the edge already exists, its
   * property is not overwritten. If u and v are not present, they are added.
   * \return the edge and bool indicating whether the edge was newly added.
   */
  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v, Object object = Object());

  /// Adds edges among all given vertices.
  void add_clique(const std::vector<Arg>& vertices);

  /// Removes a vertex from the graph and all its incident edges.
  void remove_vertex(Arg u);

  /// Removes an undirected edge {u, v}.
  void remove_edge(Arg u, Arg v);

  /// Removes all edges incident to a vertex.
  void remove_edges(Arg u);

  /// Removes all edges from the graph.
  void remove_edges();

  /// Removes all vertices and edges from the graph.
  void clear();

  struct EliminationStrategy {
    virtual ptrdiff_t priority(Arg u, const MarkovNetwork& g) const = 0;
    virtual void update(Arg u, const MarkovNetwork& g, std::vector<Arg>& output) const = 0;
  };

  /**
   * Runs the vertex elimination algorithm on a graph. The algorithm eliminates
   * each node from the graph; eliminating a node involves connecting the node's
   * neighbors into a new clique and then removing the node from the graph.
   * The nodes are eliminated greedily in the order specified by the elimination
   * strategy.
   */
  void eliminate(EliminationStrategy& strategy, VertexVisitor visitor);

  // Implementation of edge iterator
  //--------------------------------------------------------------------------
public:
  class edge_iterator
    : public std::iterator<std::forward_iterator_tag, edge_type> {
  public:
    using reference = edge_type;
    using outer_iterator = typename VertexDataMap::const_iterator;
    using inner_iterator = typename neighbor_map::const_iterator;

    edge_iterator() {}

    edge_iterator(outer_iterator it1, outer_iterator end1)
      : it1_(it1), end1_(end1) {
      find_next();
    }

    edge_type operator*() const {
      return edge_type(it1_->first, it2_->first, it2_->second);
    }

    edge_iterator& operator++() {
      do {
        ++it2_;
      } while (it2_ != it1_->second.neighbors.end() &&
                it2_->first < it1_->first);
      if (it2_ == it1_->second.neighbors.end()) {
        ++it1_;
        find_next();
      }
      return *this;
    }

    edge_iterator operator++(int) {
      edge_iterator copy = *this;
      operator++();
      return copy;
    }

    bool operator==(const edge_iterator& o) const {
      return
        (it1_ == end1_ && o.it1_ == o.end1_) ||
        (it1_ == o.it1_ && it2_ == o.it2_);
    }

    bool operator!=(const edge_iterator& other) const {
      return !(operator==(other));
    }

  private:
    /// find the next non-empty neighbor map with it1_->firstt <= it2_->first
    void find_next() {
      while (it1_ != end1_) {
        it2_ = it1_->second.neighbors.begin();
        while (it2_ != it1_->second.neighbors.end() &&
                it2_->first < it1_->first) {
          ++it2_;
        }
        if (it2_ != it1_->second.neighbors.end()) {
          break;
        } else {
          ++it1_;
        }
      }
    }

    outer_iterator it1_;  ///< the iterator to the vertex data
    outer_iterator end1_; ///< the iterator past the last vertex data
    inner_iterator it2_;  ///< the iterator to the current neighbor

  }; // class edge_iterator

}; // class MarkovNetwork

/**
 * A class that represents a Markov network with strongly typed vertex and edge properties.
 *
 * \tparam VP
 *         The type of data associated with vertices. Must be void or a subclass of Object.
 * \tparam EP
 *         The type of data associated with edges. Must be void or a subclass of Object.
 *
 * \ingroup graph_types
 */
template <typename VP = void, typename EP = void>
struct MarkovNetworkT : PropertyCaster<MarkovNetwork, VP, EP> {
  bool add_vertex(Arg u, Nullable<VP> vp = Nullable<VP>()) {
    return MarkovNetwork::add_vertex(u, std::move(vp));
  }

  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v, Nullable<EP> ep = Nullable<EP>()) {
    return MarkovNetwork::add_edge(u, v, std::move(ep));
  }
};

} // namespace libgm
