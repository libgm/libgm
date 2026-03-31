#pragma once

#include <libgm/graph/undirected_edge.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/iterator/bind1_iterator.hpp>
#include <libgm/iterator/bind2_iterator.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <vector>

namespace libgm {

/**
 * A compact undirected graph stored as sorted adjacency vectors.
 *
 * The graph stores no properties and represents each vertex by its integer
 * index. `adjacency_[i]` is the sorted list of encoded vertices adjacent to
 * `i`, always followed by `null_vertex()`. Present edges are encoded as
 * `(v << 1) | 1`, erased edges as `v << 1`, and the sentinel is intentionally
 * treated as a present entry so iteration naturally stops there.
 */
class VectorGraph {
public:
  class adjacency_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = size_t;
    using pointer = void;

    adjacency_iterator() = default;

    explicit adjacency_iterator(std::vector<size_t>::const_iterator it)
      : it_(it) {
      skip_erased();
    }

    size_t operator*() const {
      return *it_ >> 1;
    }

    adjacency_iterator& operator++() {
      assert(*it_ != null_vertex());
      ++it_;
      skip_erased();
      return *this;
    }

    adjacency_iterator operator++(int) {
      adjacency_iterator result(*this);
      ++(*this);
      return result;
    }

    friend bool operator==(const adjacency_iterator& a,
                           const adjacency_iterator& b) {
      return a.it_ == b.it_;
    }

  private:
    void skip_erased() {
      while (is_erased(*it_)) {
        ++it_;
      }
    }

    std::vector<size_t>::const_iterator it_;
  };

  // Descriptors
  //--------------------------------------------------------------------------
  using vertex_descriptor = size_t;
  using edge_descriptor = UndirectedEdge<size_t>;

  // Iterators
  //--------------------------------------------------------------------------
  using out_edge_iterator =
    Bind1Iterator<adjacency_iterator, edge_descriptor, size_t>;
  using in_edge_iterator =
    Bind2Iterator<adjacency_iterator, edge_descriptor, size_t>;
  using vertex_iterator = boost::counting_iterator<size_t>;

  // Graph categories
  //--------------------------------------------------------------------------
  using directed_category = boost::undirected_tag;
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
  using VertexVisitor = std::function<void(size_t)>;

  /// Creates a graph with `count` isolated vertices.
  explicit VectorGraph(size_t count = 0)
    : adjacency_(count, std::vector<size_t>(1, null_vertex())) {}

  /// Returns the sentinel used to terminate non-empty adjacency lists.
  static size_t null_vertex() {
    return static_cast<size_t>(-1);
  }

  // Accessors
  //--------------------------------------------------------------------------
  /// Returns the edges outgoing from a vertex.
  std::ranges::subrange<out_edge_iterator> out_edges(size_t u) const;

  /// Returns the edges incoming to a vertex.
  std::ranges::subrange<in_edge_iterator> in_edges(size_t u) const;

  /// Returns the vertices adjacent to `u`.
  std::ranges::subrange<adjacency_iterator> adjacent_vertices(size_t u) const;

  /// Returns the range of all vertices.
  std::ranges::subrange<vertex_iterator> vertices() const {
    return {vertex_iterator(0), vertex_iterator(adjacency_.size())};
  }

  /// Returns true if the graph has no vertices.
  bool empty() const {
    return adjacency_.empty();
  }

  /// Returns true if the graph contains the given vertex.
  bool contains(size_t u) const {
    return u < adjacency_.size();
  }

  /// Returns true if the graph contains the undirected edge `{u, v}`.
  bool contains(size_t u, size_t v) const {
    if (!contains(u) || !contains(v)) {
      return false;
    }
    return std::ranges::binary_search(adjacency_[u], encode_present(v));
  }

  /// Returns true if the graph contains the given edge.
  bool contains(edge_descriptor e) const {
    return contains(e.source(), e.target());
  }

  /// Returns the edge `{u, v}` if it exists and a null edge otherwise.
  edge_descriptor edge(size_t u, size_t v) const {
    return contains(u, v) ? edge_descriptor(u, v) : edge_descriptor();
  }

  /// Returns the number of edges adjacent to `u`.
  size_t out_degree(size_t u) const {
    return std::distance(adjacent_vertices(u).begin(), adjacent_vertices(u).end());
  }

  /// Returns the number of edges adjacent to `u`.
  size_t in_degree(size_t u) const {
    return out_degree(u);
  }

  /// Returns the number of edges adjacent to `u`.
  size_t degree(size_t u) const {
    return out_degree(u);
  }

  /// Returns the number of vertices.
  size_t num_vertices() const {
    return adjacency_.size();
  }

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds an isolated vertex and returns its descriptor.
  size_t add_vertex() {
    adjacency_.emplace_back(1, null_vertex());
    return adjacency_.size() - 1;
  }

  /// Adds edges among all specified vertices. All vertices must already exist.
  void add_clique(std::vector<size_t> vertices);

  /// Removes all edges incident to `u` and returns the number removed.
  size_t clear_vertex(size_t u);

  /// Removes the edge `{u, v}`. Returns 1 if removed and 0 otherwise.
  size_t remove_edge(size_t u, size_t v);

  /**
   * Eliminates vertices greedily according to the specified strategy.
   * Eliminating a vertex forms a clique over its neighbors and then removes
   * all incident edges.
   */
  void eliminate(const EliminationStrategy& strategy, VertexVisitor visitor);

private:
  static size_t encode_present(size_t v) {
    return (v << 1) | 1;
  }

  static size_t encode_erased(size_t v) {
    return v << 1;
  }

  static bool is_present(size_t v) {
    return (v & 1) != 0;
  }

  static bool is_erased(size_t v) {
    return (v & 1) == 0;
  }

  void mark_erased(size_t u, size_t v);

  std::vector<std::vector<size_t>> adjacency_;
};

} // namespace libgm
