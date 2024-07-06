#ifndef LIBGM_REGION_GRAPH_HPP
#define LIBGM_REGION_GRAPH_HPP

#include <libgm/argument/annotated.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/set_index.hpp>
#include <libgm/functional/output_iterator_assign.hpp>
#include <libgm/graph/algorithm/ancestors.hpp>
#include <libgm/graph/algorithm/descendants.hpp>
#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/graph/util/id.hpp>
#include <libgm/graph/vertex_traits.hpp>
#include <libgm/graph/util/void.hpp>

#include <algorithm>

namespace libgm {

  /**
   * This class represents a region graph, see Yedidia 2005.
   * Each vertex in the graph (a std::size_t) is associated with a region
   * and a counting number.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam VertexProperty
   *         A type of values stored at each vertex.
   *         Must be DefaultConstructible and CopyConstructible.
   * \tparam EdgeProperty
   *         A type of values stored at each edge.
   *         Must be DefaultConstructible and CopyConstructible.
   *
   * \ingroup graph_types
   */
  template <typename Arg,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class region_graph {

    // Forward declarations
    struct vertex_info;
    struct edge_info;

    //! The underlying graph type.
    using graph_type = directed_graph<id_t, vertex_info, edge_info>;

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Vertex, edge, and properties
    using argument_type   = Arg;
    using vertex_type     = id_t;
    using edge_type       = directed_edge<id_t>;
    using vertex_property = VertexProperty;
    using edge_property   = EdgeProperty;

    // Iterators
    using argument_iterator =
      typename set_index<id_t, domain<Arg> >::value_iterator;
    using vertex_iterator   = typename graph_type::vertex_iterator;
    using neighbor_iterator = typename graph_type::neighbor_iterator;
    using edge_iterator     = typename graph_type::edge_iterator;
    using in_edge_iterator  = typename graph_type::in_edge_iterator;
    using out_edge_iterator = typename graph_type::out_edge_iterator;

    // Constructors and basic member functions
    //--------------------------------------------------------------------------
  public:
    //! Constructs an empty region graph with no clusters.
    region_graph()
      : max_id_(0) { }

    //! Swaps two region graphs in-place.
    friend void swap(region_graph& a, region_graph& b) {
      swap(a.graph_, b.graph_);
      swap(a.cluster_index_, b.cluster_index_);
      std::swap(a.max_id_, b.max_id_);
    }

    //! Prints a human-readable representation of the region graph to stream.
    friend std::ostream&
    operator<<(std::ostream& out, const region_graph& g) {
      out << g.graph_;
      return out;
    }

    // Graph accessors
    //--------------------------------------------------------------------------

    //! Returns the union of all the clusters in this graph.
    iterator_range<argument_iterator> arguments() const {
      return cluster_index_.values();
    }

    //! Returns all edges in the graph
    iterator_range<edge_iterator>
    edges() const {
      return graph_.edges();
    }

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return graph_.vertices();
    }

    //! Returns the parents of u.
    iterator_range<neighbor_iterator>
    parents(id_t u) const {
      return graph_.parents(u);
    }

    //! Returns the children of u.
    iterator_range<neighbor_iterator>
    children(id_t u) const {
      return graph_.children(u);
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

    //! Returns true if the graph contains an undirected edge
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

    //! Returns true if the graph has no vertices
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

    //! Given a directed edge (u, v), returns a directed edge (v, u)
    //! The edge (v, u) must exist.
    edge_type reverse(const edge_type& e) const {
      return graph_.reverse(e);
    }

    //! Returns the cardinality of the union of all the clusters.
    std::size_t num_arguments() const {
      return cluster_index_.num_values();
    }

    //! Returns the cluster associated with a vertex.
    const domain<Arg>& cluster(id_t v) const {
      return graph_[v].property.domain;
    }

    //! Returns the separator associated with an edge.
    const domain<Arg>& separator(const edge_type& e) const {
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
    const annotated<Arg, EdgeProperty>& property(const edge_type& e) const {
      return grah_[e].property;
    }

    //! Returns the counting number of a region.
    int counting(id_t v) const {
      return graph_[v].counting;
    }

    //! Returns the property associated with a vertex
    VertexProperty& operator[](id_t u) {
      return graph_[u].property.object;
    }

    //! Returns the property associated with a vertex
    const VertexProperty& operator[](id_t u) const {
      return graph_[u].property.object;
    }

    //! Returns the property associated with an edge
    EdgeProperty& operator[](const edge_type& e) {
      return graph_[e].property.object;
    }

    //! Returns the property associated with an edge
    const EdgeProperty& operator[](const edge_type& e) const {
      return graph_[e].property.object;
    }

    // Queries
    //--------------------------------------------------------------------------

    //! Returns the ancestors of one or more vertices.
    std::unordered_set<id_t>
    ancestors(const std::unordered_set<id_t>& vertices) const {
      std::unordered_set<id_t> result;
      libgm::ancestors(graph_, vertices, result);
      return result;
    }

    //! Returns the ancestors of one vertices.
    std::unordered_set<id_t> ancestors(id_t v) const {
      std::unordered_set<id_t> result;
      libgm::ancestors(graph_, v, result);
      return result;
    }

    //! Returns the descendants of one or more vertices.
    std::unordered_set<id_t>
    descendants(const std::unordered_set<id_t>& vertices) const {
      std::unordered_set<id_t> result;
      libgm::descendants(graph_, vertices, result);
      return result;
    }

    //! Returns the ancestors of one vertices.
    std::unordered_set<id_t> descendants(id_t v) const {
      std::unordered_set<id_t> result;
      libgm::descendants(graph_, v, result);
      return result;
    }

    //! Returns the vertex that covers the given domain or 0 if none.
    id_t find_cover(const domain<Arg>& dom) const {
      return cluster_index_.find_min_cover(dom);
    }

    //! Returns a root vertex that covers the given domain or 0 if none.
    //! The region graph must be valid.
    id_t find_root_cover(const domain<Arg>& dom) const {
      id_t v = cluster_index_.find_min_cover(dom);
      if (v) {
        while (in_degree(v) > 0) {
          v = *parents(v).begin(); // choose arbitrary parent
        }
      }
      return v;
    }

    // Mutating operations
    //--------------------------------------------------------------------------

    /**
     * Adds a region with the given cluster and vertex property.
     * If the vertex already exists, does not perform anything.
     * \return bool indicating whether insertion took place
     */
    bool add_region(id_t v,
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
     * Adds a new region with the given cluster and  property.
     * This function always introduces a new cluster to the graph.
     * \return the newly added vertex
     */
    id_t add_region(const domain<Arg>& cluster,
                    const VertexProperty& vp = VertexProperty()) {
      ++max_id_;
      bool inserted = graph_.add_vertex(max_id_, vertex_info(cluster, vp));
      assert(inserted);
      cluster_index_.insert(max_id_, cluster);
      return max_id_;
    }

    /**
     * Adds an edge to the graph, setting the separator to
     * intersection of the two clusters at the endpoints.
     */
    std::pair<edge_type, bool>
    add_edge(id_t u, id_t v,
             const EdgeProperty& ep = EdgeProperty()) {
      return graph_.add_edge(u, v, edge_info(cluster(u) & cluster(v), ep));
    }

    //! Removes a vertex and the associated cluster and user information.
    void remove_vertex(id_t v) {
      cluster_index_.erase(v);
      graph_.remove_vertex(v);
    }

    //! Removes an undirected edge {u, v} and the associated separator and data.
    void remove_edge(id_t u, id_t v) {
      graph_.remove_edge(u, v);
    }

    //! Removes all edges incident to a vertex.
    void remove_edges(id_t u) {
      graph_.remove_edges(u);
    }

    //! Removes all edges incoming to a vertex.
    void remove_in_edges(id_t u) {
      graph_.remove_in_edges(u);
    }

    //! Removes all edges outgoing from a vertex.
    void remove_out_edges(id_t u) {
      graph_.remove_out_edges(u);
    }

    //! Removes all edges from the graph.
    void remove_edges() {
      graph_.remove_edges();
    }

    //! Removes all vertices and edges from the graph
    void clear() {
      graph_.clear();
      cluster_index_.clear();
      max_id_ = id_t(0);
    }

    /**
     * Recomputes the counting numbers.
     * Assigns each root a counting number 1, and sets the remaining
     * clusters to satisfy the running intersection property.
     */
    void update_counting() {
      partial_order_traversal(graph_, [&](id_t v) {
          if(in_degree(v) == 0) {
            graph_[v].counting = 1;
          } else {
            int sum = 0;
            for (id_t u : ancestors(v)) {
              sum += graph_[u].counting;
            }
            graph_[v].counting = 1 - sum;
          }
        });
    }

    // Region graph constructions
    //--------------------------------------------------------------------------

    /**
     * Copies the structure of a region graph to this one.
     */
    template <typename VP, typename EP>
    void structure_from(const region_graph<Arg, VP, EP>& other) {
      clear();
      for (id_t v : other.vertices()) {
        bool added = add_region(v, other.cluster(v));
        assert(added);
        graph_[v].counting = other.counting(v);
      }
      for (edge_type e : other.edges()) {
        add_edge(e.source(), e.target());
      }
    }

    /**
     * Initializes this region graph to the Bethe free energey approximation.
     * This construction places the root clusters as roots, and creates
     * singleton clusters as children.
     *
     * \tparam Range a range of domain objects.
     */
    template <typename Range>
    void bethe(const Range& root_clusters) {
      std::unordered_map<Arg, id_t> var_region;
      clear();

      for (const domain<Arg>& cluster : root_clusters) {
        id_t r = add_region(cluster);
        for (Arg var : cluster) {
          id_t& s = var_region[var];
          if (!s) { s = var_region[var] = add_region({var}); }
          add_edge(r, s);
        }

        update_counting();
      }
    }

    /**
     * Given a collection of initial regions, initializes this region graph
     * using Kikuchi construction. This method computes regions closed under
     * the intersection. The edges are initialized based on the rule that
     * two regions u and v s.t. C_u \superset C_v are connected, provided
     * that there is no region w s.t. C_u \superset C_w \superset C_v.
     *
     * \tparam Range a range of domain objects.
     */
    template <typename Range>
    void saturated(const Range& root_clusters) {
      std::unordered_set<domain<Arg> > clusters;
      set_index<id_t, domain<Arg> > index;
      clear();

      // add the root clusters
      for (const domain<Arg>& cluster : root_clusters) {
        if (!cluster.empty() && !clusters.count(cluster)) {
          id_t r = add_region(cluster);
          index.insert(r, cluster);
          clusters.insert(cluster);
        }
      }

      // compute closure under intersections
      while (!index.empty()) {
        id_t r = index.front();
        index.intersecting_sets(cluster(r), [&](id_t s) {
            if (r != s) {
              domain<Arg> cluster = index.intersection(r, s);
              if (!clusters.count(cluster)) {
                clusters.insert(cluster);
                index.insert(add_region(cluster), cluster);
              }
            }
          });
        index.erase(r);
      }

      // add the edges in the decreasing size of the cluster
      std::vector<id_t> regions(vertices().begin(), vertices().end());
      std::sort(regions.begin(), regions.end(),
                [&](id_t r, id_t s) {
                  return cluster(r).size() > cluster(s).size();
                });
      std::unordered_set<id_t> supersets;
      for (id_t r : regions) {
        supersets.clear();
        auto out = std::inserter(supersets, supersets.end());
        cluster_index_.supersets(cluster(r), make_output_iterator_assign(out));
        supersets.erase(r);
        for (id_t s : supersets) {
          bool valid = true;
          for (id_t t : children(s)) {
            if (supersets.count(t)) { valid = false; break; }
          }
          if (valid) { add_edge(s, r); }
        }
      }

      // compute the counting numbers
      update_counting();
    }

    // Private classes
    //--------------------------------------------------------------------------
  private:
    /**
     * The information stored with each vertex of the region graph.
     */
    struct vertex_info {
      //! The vertex property annotated with the cluster.
      annotated<Arg, VertexProperty> property;

      //! The counting number.
      int counting;

      //! Default constructor. Default-initializes the property.
      vertex_info()
        : counting() { }

      //! Construct sthe vertex info with teh given cluster and property.
      vertex_info(const domain<Arg>& cluster,
                  const VertexProperty& property = VertexProperty())
        : data(cluster, property), counting() { }

      //! Outputs the vertex_info to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const vertex_info& info) {
        out << '(' << info.property << " " << info.counting << ')';
        return out;
      }
    }; // struct vertex_info

    /**
     * The information stored with each edge of the region graph.
     */
    struct edge_info {
      //! The edge property annotated with the separator.
      annotated<Arg, EdgeProperty> property;

      //! Default constructor. Default-initializes the property.
      edge_info() : property() { }

      //! Constructor
      edge_info(const domain<Arg>& separator, const EdgeProperty& property)
        : data(separator, property) { }

      //! Outputs the edge informaitno to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const edge_info& info) {
        out << '(' << info.property << ')';
        return out;
      }
    }; // struct edge_info

    // Private data members
    //--------------------------------------------------------------------------

    //! An index that maps variables (nodes) to vertices this region graph.
    set_index<id_t, domain<Arg> > cluster_index_;

    //! The underlying graph
    directed_graph<id_t, vertex_info, edge_info> graph_;

    //! The largest ID value seen so far.
    id_t max_id_;

  }; // class region_graph

} // namespace libgm

#endif
