#ifndef LIBGM_CLUSTER_GRAPH_HPP
#define LIBGM_CLUSTER_GRAPH_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/datastructure/set_index.hpp>
#include <libgm/graph/algorithm/mst.hpp>
#include <libgm/graph/algorithm/test_connected.hpp>
#include <libgm/graph/algorithm/test_tree.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/algorithm/triangulate.hpp>
#include <libgm/graph/bidirectional.hpp>
#include <libgm/graph/id.hpp>
#include <libgm/graph/property_fn.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/graph/void.hpp>

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
   * \tparam Domain the domain type that represents the clusters and separators
   * \tparam VertexProperty the type of values stored at each vertex.
   *         Must be DefaultConstructible and the CopyConstructible.
   * \tparam EdgeProperty the type of values stored at each edge.
   *         Must be DefaultConstructible and the CopyConstructible concept.
   *         should have a fast default constructor, since we temporarily
   *         create a superlinear number of edges to compute the MST.
   *
   * \ingroup graph_types
   */
  template <typename Domain,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class cluster_graph {

    // Forward declarations
    struct vertex_info;
    struct edge_info;

    //! The underlying graph type.
    typedef undirected_graph<id_t, vertex_info, edge_info> graph_type;

    // Public type declarations
    //==========================================================================
  public:
    // Vertex, edge, and properties
    typedef id_t                  vertex_type;
    typedef undirected_edge<id_t> edge_type;
    typedef VertexProperty        vertex_property;
    typedef EdgeProperty          edge_property;

    // Iterators
    typedef typename graph_type::vertex_iterator   vertex_iterator;
    typedef typename graph_type::neighbor_iterator neighbor_iterator;
    typedef typename graph_type::edge_iterator     edge_iterator;
    typedef typename graph_type::in_edge_iterator  in_edge_iterator;
    typedef typename graph_type::out_edge_iterator out_edge_iterator;

    // The domain
    typedef typename Domain::value_type argument_type;
    typedef typename argument_traits<argument_type>::hasher argument_hasher;
    typedef typename set_index<id_t, Domain, argument_hasher>::value_iterator
      argument_iterator;
    typedef std::unordered_set<argument_type, argument_hasher> argument_set;

    // Constructors and destructors
    //==========================================================================
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
      for (edge_type e : graph_.edges()) {
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
    //==========================================================================

    //! Returns the null vertex, guaranteed to be id_t().
    static id_t null_vertex() {
      return id_t();
    }

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
    bool contains(const edge_type& e) const {
      return graph_.contains(e);
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    edge_type edge(id_t u, id_t v) const {
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
    edge_type reverse(const edge_type& e) const {
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
    std::size_t count(argument_type x) const {
      return cluster_index_.count(x);
    }

    //! Returns the cluster associated with a vertex.
    const Domain& cluster(id_t v) const {
      return graph_[v].cluster;
    }

    //! Returns the separator associated with an edge.
    const Domain& separator(const edge_type& e) const {
      return graph_[e].separator;
    }

    //! Returns the separator associated with an edge.
    const Domain& separator(id_t u, id_t v) const {
      return graph_(u, v).separator;
    }

    //! Returns a pre-computed reachable set associated with a directed edge.
    const std::vector<argument_type>& reachable(const edge_type& e) const {
      return graph_[e].reachable(e);
    }

    //! Returns true if the vertex has been marked.
    bool marked(id_t v) const {
      return graph_[v].marked;
    }

    //! Returns true if the edge has been marked.
    bool marked(const edge_type& e) const {
      return graph_[e].marked;
    }

    //! Returns the property associated with a vertex.
    VertexProperty& operator[](id_t u) {
      return graph_[u].property;
    }

    //! Returns the property associated with a vertex.
    const VertexProperty& operator[](id_t u) const {
      return graph_[u].property;
    }

    //! Returns the property associated with an edge.
    EdgeProperty& operator[](const edge_type& e) {
      return graph_[e].property;
    }

    //! Returns the property associated with an edge
    const EdgeProperty& operator[](const edge_type& e) const {
      return graph_[e].property;
    }

    /**
     * Returns the property associated with edge {u, v}.
     * The edge must exist.
     */
    const EdgeProperty& operator()(id_t u, id_t v) const {
      return graph_(u, v).property;
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
    //==========================================================================

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
      for (argument_type x : cluster_index_.values()) {
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
    id_t find_cluster_cover(const Domain& domain) const {
      return cluster_index_.find_min_cover(domain);
    }

    /**
     * Returns an edge whose separator covers the supplied domain.
     * If there are multiple such edges, one with the smallest separator
     * is returned. If there is no such edge, returns a null edge.
     */
    edge_type find_separator_cover(const Domain& domain) const {
      return separator_index_.find_min_cover(domain);
    }

    /**
     * Returns a vertex whose clique cover meets (intersects) the supplied
     * domain. The returned vertex is the one that has the smallest cluster
     * that has maximal intersection with the supplied domain.
     */
    id_t find_cluster_meets(const Domain& domain) const {
      return cluster_index_.find_max_intersection(domain);
    }

    /**
     * Returns an edge whose separator meets (intersects) the supplied
     * domain. The returned edge is the one that has the smallest separator
     * that has maximal intersection with the supplied domain.
     */
    edge_type find_separator_meets(const Domain& domain) const {
      return separator_index_.find_max_intersection(domain);
    }

    /**
     * Visits the vertices whose clusters overlap the supplied domain.
     */
    void intersecting_clusters(const Domain& domain,
                               std::function<void(id_t)> visitor) const {
      cluster_index_.intersecting_sets(domain, std::move(visitor));
    }

    /**
     * Visits the edges whose separators overlap the supplied domain.
     */
    void intersecting_separators(const Domain& domain,
                                 std::function<void(edge_type)> visitor) const {
      separator_index_.intersecting_sets(domain, std::move(visitor));
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
    void compute_reachable(bool past_empty, const Domain& filter) {
      argument_set set(filter.begin(), filter.end());
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
    void mark_subtree_cover(const Domain& domain, bool force_continuous) {
      if (empty()) { return; }

      // Initialize the vertices to be white.
      for (id_t v : vertices()) {
        graph_[v].marked = false;
      }

      // Compute the reachable variables for the set.
      compute_reachable(force_continuous, domain);

      // The edges that must be in the subtree are those such that the
      // reachable variables in both directions have a non-empty
      // symmetric difference.
      argument_set cover;
      for (edge_type e : edges()) {
        id_t u = e.source();
        id_t v = e.target();
        const std::vector<argument_type>& r1 = graph_[e].reachable.forward;
        const std::vector<argument_type>& r2 = graph_[e].reachable.reverse;
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
      Domain uncovered;
      for (argument_type x : domain) {
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
    //==========================================================================

    /**
     * Adds a vertex with the given cluster and vertex property.
     * If the vertex already exists, does not perform anything.
     * \return bool indicating whether insertion took place
     */
    bool add_cluster(id_t v,
                     const Domain& cluster,
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
    id_t add_cluster(const Domain& cluster,
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
    std::pair<edge_type, bool>
    add_separator(id_t u, id_t v,
                  const Domain& separator,
                  const EdgeProperty& ep = EdgeProperty()) {
      assert(subset(separator, cluster(u)));
      assert(subset(separator, cluster(v)));
      auto result = graph_.add_edge(u, v, edge_info(separator, ep));
      if (result.second) {
        separator_index_.insert(result.first, separator);
      }
      return result;
    }

    /**
     * Adds an edge {u, v} to the graph, setting the separator to the
     * intersection of the two clusters at the endpoints.
     * If the edge already exists, doe snot perform anything.
     * \return the edge and bool indicatign whether the insertion took place
     */
    std::pair<edge_type, bool>
    add_edge(id_t u, id_t v) {
      auto result = graph_.add_edge(u, v, edge_info(cluster(u) & cluster(v)));
      if (result.second) {
        separator_index_.insert(result.first, separator(result.first));
      }
      return result;
    }

    /**
     * Updates the cluster associated with an existing vertex.
     */
    void update_cluster(id_t u, const Domain& cluster) {
      if (graph_[u].cluster != cluster) {
        cluster_index_.erase(u);
        cluster_index_.insert(u, cluster);
        graph_[u].cluster = cluster;
      }
    }

    /**
     * Updates the separator associated with an edge.
     */
    void update_separator(const edge_type& e, const Domain& separator) {
      if (graph_[e].separator != separator) {
        separator_index_.erase(e);
        separator_index_.insert(e, separator);
        graph_[e].separator = separator;
      }
    }

    /**
     * Merges two adjacent vertices and their clusters. The edge (u,v) and
     * the source vertex u are deleted, and the target vertex v is made
     * adjacent to all neighbors of u (the information on these edges is
     * copied from the original edges). The new clique of v is set as the
     * union of the original cliques of u and v, and the separators remain
     * unchanged. If the graph is a tree that satisfies the running
     * intersection property, the property will still hold after the merge.
     *
     * \return the retained vertex
     */
    id_t merge(const edge_type& e) {
      id_t u = e.source();
      id_t v = e.target();
      for (edge_type in : in_edges(u)) {
        if (in.source() != v) {
          graph_.add_edge(in.source(), v, graph_[in]);
        }
      }
      graph_[v].cluster = cluster(u) | cluster(v);
      this->remove_vertex(u);
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
      for (edge_type e : graph_.out_edges(u)) {
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
    //==========================================================================

    /**
     * Extends the cliques, so that tree satisfies the running intersection
     * property.
     */
    void triangulate() {
      compute_reachable(true);
      for (id_t v : vertices()) {
        Domain c = cluster(v);
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
      libgm::triangulate_maximal<Domain>(g, [&](Domain&& dom) {
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
      for (const Domain& clique : cliques) {
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
      std::vector<edge_type> tree_edges;
      kruskal_minimum_spanning_tree(
        graph_,
        [&](const edge_type& e) {
          return -int(cluster_index_.intersection_size(e.source(), e.target()));
        },
        std::back_inserter(tree_edges)
      );

      // Remove all edges and add back the computed edges
      graph_.remove_edges();
      for (const edge_type& e : tree_edges) {
        add_edge(e.source(), e.target());
      }
    }

    // Private classes
    //==========================================================================
  private:
    /**
     * The information stored with each vertex of the cluster graph.
     */
    struct vertex_info {
      //! The cluster associated with this vertex.
      Domain cluster;

      //! The property associated with the vertex.
      VertexProperty property;

      //! True if the vertex has been marked. This field is not serialized.
      bool marked;

      //! Default constructor. Default-initializes the property.
      vertex_info()
        : property(), marked(false) { }

      //! Constructs the vertex info with the given cluster and property.
      vertex_info(const Domain& cluster,
                  const VertexProperty& property = VertexProperty())
        : cluster(cluster), property(property), marked(false) { }

      //! Compares the cluster and vertex property stored at two vertices.
      friend bool operator==(const vertex_info& a, const vertex_info& b) {
        return a.cluster == b.cluster && a.property == b.property;
      }

      //! Compares the cluster and vertex property stored at two vertices.
      friend bool operator!=(const vertex_info& a, const vertex_info& b) {
        return a.cluster != b.cluster || a.property != b.property;
      }

      //! Serializes cluster and property.
      void save(oarchive& ar) const {
        ar << cluster << property;
      }

      //! Deserializes cluster and property.
      void load(iarchive& ar) {
        ar >> cluster >> property;
      }

      //! Outputs the vertex_info to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const vertex_info& info) {
        out << '(' << info.cluster
            << ' ' << info.property
            << ' ' << info.marked
            << ')';
        return out;
      }

    }; // struct vertex_info

    /**
     * The information stored with each edge of the cluster graph.
     */
    struct edge_info {
      //! The separator associated with the edge.
      Domain separator;

      //! The property assocaited with the edge.
      EdgeProperty property;

      //! True if the edge has been marked. This field is not serialized.
      bool marked;

      /**
       * For edge = (u, v), reachable.directed(e) stores the variables
       * in the subtree rooted at u, away from v, in the sorted order.
       * This field is not serialized.
       */
      bidirectional<std::vector<argument_type> > reachable;

      //! Default constructor. Default-initalizes the property.
      edge_info()
        : property(), marked(false) { }

      //! Constructs the edge info with the given separator and property.
      edge_info(const Domain& separator,
                const EdgeProperty& property = EdgeProperty())
        : separator(separator), property(property) { }

      //! Compares the separators and properties stored at two edges.
      friend bool operator==(const edge_info& a, const edge_info& b) {
        return a.separator == b.separator && a.property == b.property;
      }

      //! Compares the separators and properties stored at two edges.
      friend bool operator!=(const edge_info& a, const edge_info& b) {
        return a.separator != b.separator || a.property != b.property;
      }

      //! Serialize members
      void save(oarchive& ar) const {
        ar << separator << property;
      }

      //! Deserialize members
      void load(iarchive& ar) {
        ar >> separator >> property;
      }

      //! Outputs the edge information to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const edge_info& info) {
        out << '(' << info.separator
            << ' ' << info.property
            << ' ' << info.marked
            << ')';
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
                        const argument_set* filter = nullptr)
        : graph_(graph),
          propagate_past_empty_(propagate_past_empty),
          filter_(filter) { }

      void operator()(edge_type e) const {
        std::vector<argument_type> r;
        if (!graph_[e].separator.empty() || propagate_past_empty_) {
          // extract the (possibly filtered) variables from the cluster
          for (argument_type x : graph_[e.source()].cluster) {
            if (!filter_ || filter_->count(x)) { r.push_back(x); }
          }
          // compute the union of the incoming reachable variables
          for (edge_type in : graph_.in_edges(e.source())) {
            if (in.source() != e.target()) {
              const std::vector<argument_type>& rin = graph_[in].reachable(in);
              r.insert(r.end(), rin.begin(), rin.end());
            }
          }
          // eliminate duplicates
          std::sort(r.begin(), r.end());
          r.erase(std::unique(r.begin(), r.end()), r.end());
        }

        // store the result
        graph_[e].reachable(e) = r;
      }

    private:
      graph_type& graph_;
      bool propagate_past_empty_;
      const argument_set* filter_;

    }; // class reachable_visitor

    // Private data members
    //==========================================================================

    //! An index of clusters that permits fast superset/intersection queries.
    set_index<id_t, Domain, argument_hasher> cluster_index_;

    //! An index of separators that permits fast superset/intersection queries.
    set_index<edge_type, Domain, argument_hasher> separator_index_;

    //! The underlying undirected graph.
    undirected_graph<id_t, vertex_info, edge_info> graph_;

    //! The largest ID value seen so far.
    id_t max_id_;

  }; // class cluster_graph

} // namespace libgm

#endif
