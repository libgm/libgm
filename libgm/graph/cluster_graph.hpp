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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam VertexProperty
   *         A type of values stored at each vertex.
   *         Must be DefaultConstructible and the CopyConstructible.
   * \tparam EdgeProperty
   *         A type of values stored at each edge.
   *         Must be DefaultConstructible and the CopyConstructible concept.
   *         should have a fast default constructor, since we temporarily
   *         create a superlinear number of edges to compute the MST.
   *
   * \ingroup graph_types
   */
  template <typename Arg,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class cluster_graph {

    // Forward declarations
    struct vertex_info;
    struct edge_info;

    //! The underlying graph type.
    using graph_type = undirected_graph<id_t, vertex_info, edge_info>;

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Vertex type, edge type, argument_type, and properties
    using vertex_type     = id_t;
    using edge_type       = undirected_edge<id_t> ;
    using argument_type   = Arg;
    using vertex_property = VertexProperty;
    using edge_property   = EdgeProperty;

    // Iterators
    using vertex_iterator   = typename graph_type::vertex_iterator;
    using neighbor_iterator = typename graph_type::neighbor_iterator;
    using edge_iterator     = typename graph_type::edge_iterator;
    using in_edge_iterator  = typename graph_type::in_edge_iterator;
    using out_edge_iterator = typename graph_type::out_edge_iterator;
    using argument_iterator =
      typename set_index<id_t, domain<Arg> >::value_iterator;

    // Constructors and destructors
    //--------------------------------------------------------------------------
  public:
    //! Constructs an empty cluster graph with no clusters.
    cluster_graph()
      : max_id_(0) { }

    //! Swaps two cluster graphs in constant time.
    friend void swap(cluster_graph& a, cluster_graph& b) {
      swap(a.cluster_index_, b.cluster_index_);
      swap(a.separator_index_, b.separator_index_);
      swap(a.graph_, b.graph_);
      std::swap(a.max_id_, b.max_id_);
    }

    //! Serializes the cluster graph to an archive.
    void save(oarchive& ar) const {
      ar << graph_ << max_id_;
    }

    //! Deserialize the cluster graph from an archive.
    void load(iarchive& ar) {
      ar >> graph_ >> max_id_;
      cluster_index_.clear();
      separator_index_.clear();
      for (vertex_type v : graph_.vertices()) {
        cluster_index_.insert(v, cluster(v));
      }
      for (undirected_edge<id_t> e : graph_.edges()) {
        id_t u = e.source();
        id_t v = e.target();
        graph_[e].index(u, v) = cluster(u).index(separator(e));
        graph_[e].index(v, u) = cluster(v).index(separator(e));
        separator_index_.insert(e, separator(e));
      }
    }

    //! Prints a human-readable representation of the cluster graph to stream.
    friend std::ostream&
    operator<<(std::ostream& out, const cluster_graph& g) {
      out << g.graph_;
      return out;
    }

    // Graph accessors
    //--------------------------------------------------------------------------

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return graph_.vertices();
    }

    //! Returns the vertices adjacent to u.
    iterator_range<neighbor_iterator>
    neighbors(id_t u) const {
      return graph_.neighbors(u);
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return graph_.edges();
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(id_t u) const {
      return graph_.in_edges(u);
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(id_t u) const {
      return graph_.out_edges(u);
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(id_t u) const {
      return graph_.contains(u);
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(id_t u, id_t v) const {
      return graph_.contains(u, v);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(undirected_edge<id_t> e) const {
      return graph_.contains(e);
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    undirected_edge<id_t> edge(id_t u, id_t v) const {
      return graph_.edge(u, v);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t in_degree(id_t u) const {
      return graph_.in_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t out_degree(id_t u) const {
      return graph_.out_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t degree(id_t u) const {
      return graph_.degree(u);
    }

    //! Returns true if the graph has no vertices.
    bool empty() const {
      return graph_.empty();
    }

    //! Returns the number of vertices
    std::size_t num_vertices() const {
      return graph_.num_vertices();
    }

    //! Returns the number of edges
    std::size_t num_edges() const {
      return graph_.num_edges();
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
    undirected_edge<id_t> reverse(undirected_edge<id_t> e) const {
      return e.reverse();
    }

    //! Returns the union of all the clusters in this graph.
    iterator_range<argument_iterator> arguments() const {
      return cluster_index_.values();
    }

    //! Returns the cardinality of the union of all the clusters.
    std::size_t num_arguments() const {
      return cluster_index_.num_values();
    }

    //! Returns true if the number of clusters in which the argument occurs.
    std::size_t count(Arg x) const {
      return cluster_index_.count(x);
    }

    //! Returns the cluster associated with a vertex.
    const domain<Arg>& cluster(id_t v) const {
      return graph_[v].property.domain;
    }

    //! Returns the separator associated with an edge.
    const domain<Arg>& separator(undirected_edge<id_t> e) const {
      return graph_[e].property.domain;
    }

    //! Returns the separator associated with an edge.
    const domain<Arg>& separator(id_t u, id_t v) const {
      return graph_(u, v).property.domain;
    }

    //! Returns the annotated property associated with a vertex.
    const annotated<Arg, VertexProperty>& property(id_t u) const {
      return graph_[v].property;
    }

    //! Returns the annotated property associated with an edge.
    const annotated<Arg, EdgeProperty>& property(undirected_edge<id_t> e) const {
      return grah_[e].property;
    }

    //! Returns the index mapping from a domain to the given clique.
    uint_vector index(id_t v, const domain<Arg>& dom) const {
      return cluster(v).index(dom);
    }

    //! Returns the index mapping from a domain to the given separator.
    uint_vector index(undirected_edge<id_t> e, const domain<Arg>& dom) const {
      return separator(e).index(dom);
    }

    //! Returns the index mapping from the separator to the source clique.
    const uint_vector& source_index(undirected_edge<id_t> e) const {
      return graph_[e].index(e.source(), e.target());
    }

    //! Returns the index mapping from the separator to the target clique.
    const uint_vector& target_index(undirected_edge<id_t> e) const {
      return graph_[e].index(e.target(), e.source());
    }

    //! Returns a pre-computed reachable set associated with a directed edge.
    const domain<Arg>& reachable(undirected_edge<id_t> e) const {
      return graph_[e].reachable(e);
    }

    //! Returns true if the vertex has been marked.
    bool marked(id_t v) const {
      return graph_[v].marked;
    }

    //! Returns true if the edge has been marked.
    bool marked(undirected_edge<id_t> e) const {
      return graph_[e].marked;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](id_t u) {
      return graph_[u].property.object;
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator[](id_t u) const {
      return graph_[u].property.object;
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](undirected_edge<id_t> e) {
      return graph_[e].property.object;
    }

    //! Returns the property associated with an edge
    const EdgeProperty& operator[](undirected_edge<id_t> e) const {
      return graph_[e].property.object;
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge must exist.
     */
    const EdgeProperty& operator()(id_t u, id_t v) const {
      return graph_(u, v).property.object;
    }

    /**
     * Returns true if two cluster graphs are identical.
     * The property types must support operator!=().
     */
    bool operator==(const cluster_graph& other) const {
      return graph_ == other.graph_;
    }

    /**
     * Returns true if two cluster graphs are not identical.
     * The property types must support operator!=().
     */
    bool operator!=(const cluster_graph& other) const {
      return graph_ != other.graph_;
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Returns true if the graph is connected.
     */
    bool connected() const {
      return test_connected(graph_);
    }

    /**
     * Returns true if the cluster graph is a tree.
     */
    bool tree() const {
      return num_edges() == num_vertices() - 1 && connected();
    }

    /**
     * Returns true if the cluster graph satisfies the running intersection
     * property. A cluster graph satisfies the running intersection property
     * if, for each value, the clusters and separators containing that value
     * form a subtree.
     */
    bool running_intersection() const {
      for (Arg x : cluster_index_.values()) {
        std::size_t n = cluster_index_.count(x);
        id_t v = cluster_index_[x];
        std::size_t nreachable = test_tree(graph_, v, [&](const edge_type& e) {
            return separator(e).count(x) > 0 &&
              cluster(e.target()).count(x) > 0;
          });
        if (nreachable != n) { return false; }
      }
      return true;
    }

    /**
     * Returns true if the cluster graph represents a triangulated model,
     * i.e., is a tree and satisfies the running intersection property.
     */
    bool triangulated() const {
      return tree() && running_intersection();
    }

    /**
     * Returns the maximum clique size minus one.
     * Only meaningful when this graph is a tree.
     */
    std::ptrdiff_t tree_width() const {
      std::size_t max_size = 0;
      for (id_t v : vertices()) {
        max_size = std::max(max_size, cluster(v).size());
      }
      return std::ptrdiff_t(max_size) - 1;
    }

    /**
     * Returns a vertex whose clique covers (is a superset of) the supplied
     * domain. If there are multiple such vertices, returns the one with the
     * smallest cluster size (cardinality). If there is no such vertex,
     * then returns the null vertex.
     */
    id_t find_cluster_cover(const domain<Arg>& dom) const {
      return cluster_index_.find_min_cover(dom);
    }

    /**
     * Returns an edge whose separator covers the supplied domain.
     * If there are multiple such edges, one with the smallest separator
     * is returned. If there is no such edge, returns a null edge.
     */
    undirected_edge<id_t> find_separator_cover(const domain<Arg>& dom) const {
      return separator_index_.find_min_cover(dom);
    }

    /**
     * Returns a vertex whose clique cover meets (intersects) the supplied
     * domain. The returned vertex is the one that has the smallest cluster
     * that has maximal intersection with the supplied domain.
     */
    id_t find_cluster_meets(const domain<Arg>& dom) const {
      return cluster_index_.find_max_intersection(dom);
    }

    /**
     * Returns an edge whose separator meets (intersects) the supplied
     * domain. The returned edge is the one that has the smallest separator
     * that has maximal intersection with the supplied domain.
     */
    undirected_edge<id_t> find_separator_meets(const domain<Arg>& dom) const {
      return separator_index_.find_max_intersection(dom);
    }

    /**
     * Visits the vertices whose clusters overlap the supplied domain.
     */
    void intersecting_clusters(const domain<Arg>& dom,
                               std::function<void(id_t)> visitor) const {
      cluster_index_.intersecting_sets(dom, std::move(visitor));
    }

    /**
     * Visits the edges whose separators overlap the supplied domain.
     */
    void intersecting_separators(const domain<Arg>& dom,
                                 std::function<void(edge_type)> visitor) const {
      separator_index_.intersecting_sets(dom, std::move(visitor));
    }

    /**
     * Computes the reachable nodes for each directed edge in the
     * cluster graph. Requires the graph to be a tree.
     *
     * \param propagate_past_empty
     *        If set to false, then edges with empty separators
     *        are ignored. Their reachable sets are assigned empty sets.
     */
    void compute_reachable(bool past_empty) {
      mpp_traversal(graph_, id_t(), reachable_visitor(graph_, past_empty));
    }

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
    void compute_reachable(bool past_empty, const domain<Arg>& filter) {
      std::unordered_set<Arg> set(filter.begin(), filter.end());
      mpp_traversal(graph_, id_t(), reachable_visitor(graph_, past_empty, &set));
    }

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
    void mark_subtree_cover(const domain<Arg>& dom, bool force_continuous) {
      if (empty()) { return; }

      // Initialize the vertices to be white.
      for (id_t v : vertices()) {
        graph_[v].marked = false;
      }

      // Compute the reachable variables for the set.
      compute_reachable(force_continuous, dom);

      // The edges that must be in the subtree are those such that the
      // reachable variables in both directions have a non-empty
      // symmetric difference.
      std::unordered_set<Arg> cover;
      for (undirected_edge<id_t> e : edges()) {
        id_t u = e.source();
        id_t v = e.target();
        const domain<Arg>& r1 = graph_[e].reachable.forward;
        const domain<Arg>& r2 = graph_[e].reachable.reverse;
        if (!std::includes(r1.begin(), r1.end(), r2.begin(), r2.end()) &&
            !std::includes(r2.begin(), r2.end(), r1.begin(), r1.end())) {
          graph_[e].marked = true;
          graph_[u].marked = true;
          graph_[v].marked = true;
          cover.insert(cluster(u).begin(), cluster(u).end());
          cover.insert(cluster(v).begin(), cluster(v).end());
        } else {
          graph_[e].marked = false;
        }
      }

      // We must also mark vertices that are part of the subtree but
      // are not attached to any other vertex in the subtree.
      // If force_continuous = true, then either all nodes were covered
      // in the previous stage, or the nodes are contained in a single clique
      domain<Arg> uncovered;
      for (Arg x : domain) {
        if (!cover.count(x)) {
          uncovered.insert(uncovered.end(), x);
        }
      }
      while (!uncovered.empty()) {
        id_t v = find_cluster_meets(uncovered);
        assert(v);
        uncovered = uncovered - cluster(v);
        graph_[v].marked = true;
      }
    }

    // Mutating operations
    //--------------------------------------------------------------------------

    /**
     * Adds a vertex with the given cluster and vertex property.
     * If the vertex already exists, does not perform anything.
     * \return bool indicating whether insertion took place
     */
    bool add_cluster(id_t v,
                     const domain<Arg>& cluster,
                     const VertexProperty& vp = VertexProperty()) {
      max_id_ = std::max(max_id_, v);
      if (graph_.add_vertex(v, vertex_info(cluster, vp))) {
        cluster_index_.insert(v, cluster);
        return true;
      } else {
        return false;
      }
    }

    /**
     * Adds a new cluster with the given propperty and returns its vertex.
     * This function always introduces a new cluster to the graph.
     */
    id_t add_cluster(const domain<Arg>& cluster,
                     const VertexProperty& vp = VertexProperty()) {
      ++max_id_;
      bool inserted = graph_.add_vertex(max_id_, vertex_info(cluster, vp));
      assert(inserted);
      cluster_index_.insert(max_id_, cluster);
      return max_id_;
    }

    /**
     * Adds an edge {u, v} to the graph with the given separator. The edge
     * endpoints must exist, and the separator must be a subset of the clusters
     * at these vertices. If the edge already exists, does not perform anything.
     * \return the edge and bool indicating whether the insertion took place
     */
    std::pair<undirected_edge<id_t>, bool>
    add_separator(id_t u, id_t v,
                  const domain<Arg>& separator,
                  const EdgeProperty& ep = EdgeProperty()) {
      assert(subset(separator, cluster(u)));
      assert(subset(separator, cluster(v)));
      auto result = graph_.add_edge(u, v, edge_info(separator, ep));
      if (result.second) {
        separator_index_.insert(result.first, separator);
        graph_[result.first].index(u, v) = cluster(u).index(separator);
        graph_[result.first].index(v, u) = cluster(v).index(separator);
      }
      return result;
    }

    /**
     * Adds an edge {u, v} to the graph, setting the separator to the
     * intersection of the two clusters at the endpoints.
     * If the edge already exists, does not perform anything.
     * \return the edge and bool indicatign whether the insertion took place
     */
    std::pair<undirected_edge<id_t>, bool>
    add_edge(id_t u, id_t v) {
      return add_separator(u, v, edge_info(cluster(u) & cluster(v)));
    }

    /**
     * Updates the cluster associated with an existing vertex.
     */
    void update_cluster(id_t u, const domain<Arg>& cluster) {
      if (graph_[u].property.domain != cluster) {
        graph_[u].property.domain = cluster;
        cluster_index_.erase(u);
        cluster_index_.insert(u, cluster);
      }
    }

    /**
     * Updates the separator associated with an edge.
     */
    void update_separator(undirected_edge<id_t> e, const domain<Arg>& sep) {
      if (separator(e) != sep) {
        id_t u = e.source();
        id_t v = e.target();
        graph_[e].property.domain = sep;
        graph_[e].index(u, v) = cluster(u).index(sep);
        graph_[e].index(v, u) = cluster(v).index(sep);
        separator_index_.erase(e);
        separator_index_.insert(e, sep);
      }
    }

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
    id_t merge(undirected_edge<id_t> e) {
      id_t u = e.source();
      id_t v = e.target();
      graph_[v].cluster += cluster(u);
      for (undirected_edge<id_t> in : in_edges(u)) {
        if (in.source() != v) {
          auto pair = graph_.add_edge(v, in.source(), graph_[in]);
          assert(pair.second);
          undirected_edge<id_t> e = pair.first;
          graph_[e].index(v, in.source()) = cluster(v).index(separator(e));
        }
      }
      remove_vertex(u);
      return v;
    }

    //! Removes a vertex and the associated cluster and property.
    void remove_vertex(id_t v) {
      cluster_index_.erase(v);
      graph_.remove_vertex(v);
    }

    //! Removes an undirected edge {u, v} and the associated separator and data.
    void remove_edge(id_t u, id_t v) {
      separator_index_.erase(graph_.edge(u, v));
      graph_.remove_edge(u, v);
    }

    //! Removes all edges incindent to a vertex
    void remove_edges(id_t u) {
      for (undirected_edge<id_t> e : graph_.out_edges(u)) {
        separator_index_.erase(e);
      }
      graph_.remove_edges(u);
    }

    //! Removes all edges from the graph
    void remove_edges() {
      graph_.remove_edges();
      separator_index_.clear();
    }

    //! Removes all vertices and edges from the graph
    void clear() {
      graph_.clear();
      cluster_index_.clear();
      separator_index_.clear();
      max_id_ = id_t(0);
    }

    // Triangulation
    //--------------------------------------------------------------------------

    /**
     * Extends the cliques, so that tree satisfies the running intersection
     * property.
     */
    void triangulate() {
      compute_reachable(true);
      for (id_t v : vertices()) {
        domain<Arg> c = cluster(v);
        in_edge_iterator it1, end;
        for (std::tie(it1, end) = in_edges(v); it1 != end; ++it1) {
          in_edge_iterator it2 = it1;
          while (++it2 != end) {
            std::set_intersection(reachable(*it1).begin(), reachable(*it1).end(),
                                  reachable(*it2).begin(), reachable(*it2).end(),
                                  std::inserter(c, c.end()));
          }
        }
        c.unique();
        graph_[v].cluster = c;
      }
      assert(false); // TODO: update the separators
    }

    /**
     * Initializes this cluster graph to a junction tree that represents the
     * triangulation of the supplied graph.
     *
     * \tparam Graph an undirected graph type whose vertices correspond to
     *               variables stored in Domain.
     * \tparam Strategy a type that models the EliminationStrategy concept
     */
    template <typename Graph, typename Strategy>
    void triangulated(Graph& g, Strategy strategy) {
      clear();
      libgm::triangulate_maximal<Domain>(g, [&](domain<Arg>&& dom) {
          add_cluster(dom); }, strategy);
      mst_edges();
    }

    /**
     * Initializes the cluster graph to a junction tree with the given
     * cliques of a triangulated greaph.
     *
     * \tparam Range a range with values convertible to Domain
     */
    template <typename Range>
    void triangulated(const Range& cliques) {
      clear();
      for (const domain<Arg>& clique : cliques) {
        add_cluster(clique);
      }
      mst_edges();
    }

    /**
     * Initializes the edges using the maximum spanning tree algorithm based
     * on the separator sizes associated with each edge. This guarantees that
     * if the clusters are triangulated to begin with, the cluster graph will
     * be a junction tree.
     */
    void mst_edges() {
      remove_edges();
      if (empty()) { return; }

      // Select a distinguished vertex of the tree.
      id_t root = *vertices().begin();

      // For each pair of overlapping cliques, add a candidate edge to the graph
      // Also, add edges between a distinguished vertex and all other vertices,
      // to ensure that the resulting junction tree is connected
      for (id_t u : vertices()) {
        intersecting_clusters(cluster(u), [&](id_t v) {
            if (u < v) { graph_.add_edge(u, v); }
          });
        if (root != u) { graph_.add_edge(root, u); }
      }

      // Compute the edges of a maximum spanning tree using Kruskal's algorithm
      std::vector<undirected_edge<id_t>> tree_edges;
      kruskal_minimum_spanning_tree(
        graph_,
        [&](undirected_edge<id_t> e) {
          return -int(cluster_index_.intersection_size(e.source(), e.target()));
        },
        std::back_inserter(tree_edges)
      );

      // Remove all edges and add back the computed edges
      graph_.remove_edges();
      for (undirected_edge<id_t> e : tree_edges) {
        add_edge(e.source(), e.target());
      }
    }

    // Private classes
    //--------------------------------------------------------------------------
  private:
    /**
     * The information stored with each vertex of the cluster graph.
     */
    struct vertex_info {
      //! The vertex property with the cluster.
      annotated<Arg, VertexProperty> property;

      //! True if the vertex has been marked. This field is not serialized.
      bool marked;

      //! Default constructor. Default-initializes the property.
      vertex_info()
        : marked(false) { }

      //! Constructs the vertex info with the given cluster and property.
      vertex_info(const domain<Arg>& cluster,
                  const VertexProperty& property = VertexProperty())
        : data(cluster, property), marked(false) { }

      //! Compares the cluster and vertex property stored at two vertices.
      friend bool operator==(const vertex_info& a, const vertex_info& b) {
        return a.property == b.property;
      }

      //! Compares the cluster and vertex property stored at two vertices.
      friend bool operator!=(const vertex_info& a, const vertex_info& b) {
        return a.property != b.property;
      }

      //! Serializes cluster and property.
      void save(oarchive& ar) const {
        ar << data;
      }

      //! Deserializes cluster and property.
      void load(iarchive& ar) {
        ar >> data;
      }

      //! Outputs the vertex_info to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const vertex_info& info) {
        out << '(' << info.property << ' ' << info.marked << ')';
        return out;
      }

    }; // struct vertex_info

    /**
     * The information stored with each edge of the cluster graph.
     */
    struct edge_info {
      //! The edge property annotated with the separator.
      annotated<Arg, EdgeProperty> property;

      //! The dimensions of the separator in each respective clique.
      bidirectional<uint_vector> index;

      /**
       * For edge = (u, v), reachable.directed(e) stores the variables
       * in the subtree rooted at u, away from v, in the sorted order.
       * This field is not serialized.
       */
      bidirectional<domain<Arg> > reachable;

      //! True if the edge has been marked. This field is not serialized.
      bool marked;

      //! Default constructor. Default-initalizes the property.
      edge_info()
        : property(), marked(false) { }

      //! Constructs the edge info with the given separator and property.
      edge_info(const domain<Arg>& separator,
                const EdgeProperty& property = EdgeProperty())
        : data(separator, property), marked(false) { }

      //! Compares the separators and properties stored at two edges.
      friend bool operator==(const edge_info& a, const edge_info& b) {
        return a.property == b.property;
      }

      //! Compares the separators and properties stored at two edges.
      friend bool operator!=(const edge_info& a, const edge_info& b) {
        return a.property != b.property;
      }

      //! Serialize members
      void save(oarchive& ar) const {
        ar << data;
      }

      //! Deserialize members
      void load(iarchive& ar) {
        ar >> data;
      }

      //! Outputs the edge information to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const edge_info& info) {
        out << '(' << info.property << ' ' << info.marked << ')';
        return out;
      }

    }; // struct edge_info

    /**
     * An edge visitor that computes the reachable vars in the cluster graph.
     * When using with a message passing protocol (MPP) traversal on a tree,
     * this visitor is guaranteed to compute the reachable variables for each
     * edge the tree.
     */
    class reachable_visitor {
    public:
      reachable_visitor(graph_type& graph,
                        bool propagate_past_empty,
                        const std::unordered_set<Arg>* filter = nullptr)
        : graph_(graph),
          propagate_past_empty_(propagate_past_empty),
          filter_(filter) { }

      void operator()(undirected_edge<id_t> e) const {
        domain<Arg> r;
        if (!graph_[e].property.domain.empty() || propagate_past_empty_) {
          // extract the (possibly filtered) variables from the cluster
          for (Arg x : graph_[e.source()].cluster) {
            if (!filter_ || filter_->count(x)) { r.push_back(x); }
          }
          // compute the union of the incoming reachable variables
          for (undirected_edge<id_t> in : graph_.in_edges(e.source())) {
            if (in.source() != e.target()) {
              r.append(graph_[in].reachable(in));
            }
          }
          // eliminate duplicates
          r.unique();
        }

        // store the result
        graph_[e].reachable(e) = r;
      }

    private:
      graph_type& graph_;
      bool propagate_past_empty_;
      const std::unordered_set<Arg>* filter_;

    }; // class reachable_visitor

    // Private data members
    //--------------------------------------------------------------------------

    //! An index of clusters that permits fast superset/intersection queries.
    set_index<id_t, domain<Arg> > cluster_index_;

    //! An index of separators that permits fast superset/intersection queries.
    set_index<edge_type, domain<Arg> > separator_index_;

    //! The underlying undirected graph.
    undirected_graph<id_t, vertex_info, edge_info> graph_;

    //! The largest ID value seen so far.
    id_t max_id_;

  }; // class cluster_graph

} // namespace libgm

#endif
