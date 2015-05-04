#ifndef LIBGM_REGION_GRAPH_HPP
#define LIBGM_REGION_GRAPH_HPP

#include <libgm/global.hpp>
#include <libgm/datastructure/set_index.hpp>
#include <libgm/functional/output_iterator_assign.hpp>
#include <libgm/graph/algorithm/ancestors.hpp>
#include <libgm/graph/algorithm/descendants.hpp>
#include <libgm/graph/algorithm/graph_traversal.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/graph/property_fn.hpp>

#include <algorithm>

namespace libgm {

  /**
   * This class represents a region graph, see Yedidia 2005.
   * Each vertex in the graph (a size_t) is associated with a region
   * and a counting number.
   *
   * \tparam Domain  the domain type that represents regions
   * \tparam VertexProperty the property associated with a vertex.
   *         Models the DefaultConstructible and CopyConstructible concepts.
   * \tparam EdgeProperty the property associated with an edge.
   *         Models the DefaultConstructible and CopyConstructible concepts.
   *
   * \ingroup graph_types
   */
  template <typename Domain,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class region_graph {

    // Forward declarations
    struct vertex_info;
    struct edge_info;

    //! The underlying graph type.
    typedef directed_graph<size_t, vertex_info, edge_info> graph_type;

    // Public type declarations
    // =========================================================================
  public:
    // vertex, edge, and propertiies
    typedef size_t                vertex_type;
    typedef directed_edge<size_t> edge_type;
    typedef VertexProperty        vertex_property;
    typedef EdgeProperty          edge_property;
  
    // iterators
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  neighbor_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;

    // The domain
    typedef typename Domain::value_type value_type;

    // Constructors and basic member functions
    // =========================================================================
  public:
    //! Constructs an empty region graph with no clusters.
    region_graph()
      : next_vertex_(1) { }
  
    //! Swaps two region graphs in-place.
    friend void swap(region_graph& a, region_graph& b) {
      swap(a.graph_, b.graph_);
      swap(a.cluster_index_, b.cluster_index_);
      std::swap(a.next_vertex_, b.next_vertex_);
    }

    //! Prints a human-readable representation of the region graph to stream.
    friend std::ostream&
    operator<<(std::ostream& out, const region_graph& g) {
      out << g.graph_;
      return out;
    }

    // Graph accessors
    // =========================================================================

    //! Returns the range of all vertices.
    iterator_range<vertex_iterator>
    vertices() const {
      return graph_.vertices();
    }

    //! Returns the parents of u.
    iterator_range<neighbor_iterator>
    parents(size_t u) const {
      return graph_.parents(u);
    }

    //! Returns the children of u.
    iterator_range<neighbor_iterator>
    children(size_t u) const {
      return graph_.children(u);
    }

    //! Returns all edges in the graph
    iterator_range<edge_iterator>
    edges() const {
      return graph_.edges();
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(size_t u) const {
      return graph_.in_edges(u);
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(size_t u) const {
      return graph_.out_edges(u);
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(size_t u) const {
      return graph_.contains(u);
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(size_t u, size_t v) const {
      return graph_.contains(u, v);
    }

    //! Returns true if the graph contains an undirected edge
    bool contains(const edge_type& e) const {
      return graph_.contains(e);
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    edge_type edge(size_t u, size_t v) const {
      return graph_.edge(u, v);
    }

    //! Returns the number of edges adjacent to a vertex.
    size_t in_degree(size_t u) const {
      return graph_.in_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    size_t out_degree(size_t u) const {
      return graph_.out_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    size_t degree(size_t u) const {
      return graph_.degree(u);
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return graph_.empty();
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return graph_.num_vertices();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return graph_.num_edges();
    }

    //! Given a directed edge (u, v), returns a directed edge (v, u)
    //! The edge (v, u) must exist.
    edge_type reverse(const edge_type& e) const { 
      return graph_.reverse(e);
    }

    //! Returns the cluster associated with a vertex.
    const Domain& cluster(size_t v) const {
      return graph_[v].cluster;
    }

    //! Returns the separator associated with an edge.
    const Domain& separator(const edge_type& e) const {
      return graph_[e].separator;
    }

    //! Returns the separator associated with an edge.
    const Domain& separator(size_t u, size_t v) const {
      return graph_(u, v).separator;
    }

    //! Returns the counting number of a region.
    int counting(size_t v) const {
      return graph_[v].counting;
    }

    //! Returns the property associated with a vertex
    VertexProperty& operator[](size_t u) {
      return graph_[u].property;
    }

    //! Returns the property associated with a vertex
    const VertexProperty& operator[](size_t u) const {
      return graph_[u].property;
    }

    //! Returns the property associated with an edge
    EdgeProperty& operator[](const edge_type& e) {
      return graph_[e].property;
    }

    //! Returns the property associated with an edge
    const EdgeProperty& operator[](const edge_type& e) const {
      return graph_[e].property;
    }

    // Queries
    //==========================================================================

    //! Returns the ancestors of one or more vertices.
    std::unordered_set<size_t>
    ancestors(const std::unordered_set<size_t>& vertices) const {
      std::unordered_set<size_t> result;
      libgm::ancestors(graph_, vertices, result);
      return result;
    }

    //! Returns the ancestors of one vertices.
    std::unordered_set<size_t> ancestors(size_t v) const {
      std::unordered_set<size_t> result;
      libgm::ancestors(graph_, v, result);
      return result;
    }

    //! Returns the descendants of one or more vertices.
    std::unordered_set<size_t>
    descendants(const std::unordered_set<size_t>& vertices) const {
      std::unordered_set<size_t> result;
      libgm::descendants(graph_, vertices, result);
      return result;
    }

    //! Returns the ancestors of one vertices.
    std::unordered_set<size_t> descendants(size_t v) const {
      std::unordered_set<size_t> result;
      libgm::descendants(graph_, v, result);
      return result;
    }

    //! Returns the vertex that covers the given domain or 0 if none.
    size_t find_cover(const Domain& domain) const {
      return cluster_index_.find_min_cover(domain);
    }
    
    //! Returns a root vertex that covers the given domain or 0 if none.
    //! The region graph must be valid.
    size_t find_root_cover(const Domain& domain) const {
      size_t v = cluster_index_.find_min_cover(domain);
      if (v) {
        while (in_degree(v) > 0) {
          v = *parents(v).begin(); // choose arbitrary parent
        }
      }
      return v;
    }

    // Mutating operations
    //==========================================================================

    /**
     * Adds a region with the given cluster and vertex property.
     * If the vertex already exists, does not perform anything.
     * \return bool indicating whether insertion took place
     */
    bool add_region(size_t v,
                    const Domain& cluster,
                    const VertexProperty& vp = VertexProperty()) {
      next_vertex_ = std::max(next_vertex_, v + 1);
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
    size_t add_region(const Domain& cluster, 
                      const VertexProperty& vp = VertexProperty()) {
      bool inserted = graph_.add_vertex(next_vertex_, vertex_info(cluster, vp));
      assert(inserted);
      cluster_index_.insert(next_vertex_, cluster);
      return next_vertex_++;
    }

    /**
     * Adds an edge to the graph, setting the separator to
     * intersection of the two clusters at the endpoints.
     */
    std::pair<edge_type, bool>
    add_edge(size_t u, size_t v, const EdgeProperty& ep = EdgeProperty()) {
      return graph_.add_edge(u, v, edge_info(cluster(u) & cluster(v), ep));
    }

    //! Removes a vertex and the associated cluster and user information.
    void remove_vertex(size_t v) {
      cluster_index_.erase(v);
      graph_.remove_vertex(v);
    }

    //! Removes an undirected edge {u, v} and the associated separator and data.
    void remove_edge(size_t u, size_t v) {
      graph_.remove_edge(u, v);
    }

    //! Removes all edges incident to a vertex.
    void remove_edges(size_t u) {
      graph_.remove_edges(u);
    }

    //! Removes all edges incoming to a vertex.
    void remove_in_edges(size_t u) {
      graph_.remove_in_edges(u);
    }

    //! Removes all edges outgoing from a vertex.
    void remove_out_edges(size_t u) {
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
      next_vertex_ = 1;
    }

    /**
     * Recomputes the counting numbers.
     * Assigns each root a counting number 1, and sets the remaining
     * clusters to satisfy the running intersection property.
     */
    void update_counting() {
      partial_order_traversal(graph_, [&](size_t v) {
          if(in_degree(v) == 0) {
            graph_[v].counting = 1;
          } else {
            int sum = 0;
            for (size_t u : ancestors(v)) {
              sum += graph_[u].counting;
            }
            graph_[v].counting = 1 - sum;
          }
        });
    }
    
    // Region graph constructions
    //==========================================================================

    /**
     * Copies the structure of a region graph to this one.
     */
    template <typename VP, typename EP>
    void structure_from(const region_graph<Domain, VP, EP>& other) {
      clear();
      for (size_t v : other.vertices()) {
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
      std::unordered_map<value_type, size_t> var_region; // singleton regions
      clear();

      for (const Domain& cluster : root_clusters) {
        size_t r = add_region(cluster);
        for (value_type var : cluster) {
          size_t& s = var_region[var];
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
      std::unordered_set<Domain> clusters;
      set_index<size_t, Domain> index;
      clear();

      // add the root clusters
      for (const Domain& cluster : root_clusters) {
        if (!cluster.empty() && !clusters.count(cluster)) {
          size_t r = add_region(cluster);
          index.insert(r, cluster);
          clusters.insert(cluster);
        }
      }

      // compute closure under intersections
      while (!index.empty()) {
        size_t r = index.front();
        index.intersecting_sets(cluster(r), [&](size_t s) {
            if (r != s) {
              Domain cluster = index.intersection(r, s);
              if (!clusters.count(cluster)) {
                clusters.insert(cluster);
                index.insert(add_region(cluster), cluster);
              }
            }
          });
        index.erase(r);
      }

      // add the edges in the decreasing size of the cluster
      std::vector<size_t> regions(vertices().begin(), vertices().end());
      std::sort(regions.begin(), regions.end(), [&](size_t r, size_t s) {
          return cluster(r).size() > cluster(s).size();
        });
      std::unordered_set<size_t> supersets;
      for (size_t r : regions) {
        supersets.clear();
        auto out = std::inserter(supersets, supersets.end());
        cluster_index_.supersets(cluster(r), make_output_iterator_assign(out));
        supersets.erase(r);
        for (size_t s : supersets) {
          bool valid = true;
          for (size_t t : children(s)) {
            if (supersets.count(t)) { valid = false; break; }
          }
          if (valid) { add_edge(s, r); }
        }
      }

      // compute the counting numbers
      update_counting();
    }

    // Private classes
    //==========================================================================
  private:
    /**
     * The information stored with each vertex of the region graph.
     */
    struct vertex_info {
      //! The cluster associated with this vertex.
      Domain cluster;

      //! The counting number.
      int counting;

      //! The property associated with this vertex.
      VertexProperty property;

      //! Default constructor. Default-initializes the property.
      vertex_info()
        : counting(), property() { }

      //! Construct sthe vertex info with teh given cluster and property.
      vertex_info(const Domain& cluster,
                  const VertexProperty& property = VertexProperty())
        : cluster(cluster), counting(), property(property) { }
      
      //! Outputs the vertex_info to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const vertex_info& info) {
        out << '(' << info.cluster 
            << ' ' << info.counting
            << ' ' << info.property
            << ')';
        return out;
      }
    }; // struct vertex_info

    /**
     * The information stored with each edge of the region graph.
     */
    struct edge_info {
      //! The intersection of the two adjacent clusters.
      Domain separator;

      //! The property associated with the edge.
      EdgeProperty property;

      //! Default constructor. Default-initializes the property.
      edge_info() : property() { }

      //! Constructor
      edge_info(const Domain& separator, const EdgeProperty& property)
        : separator(separator), property(property) { }

      //! Outputs the edge informaitno to an output stream.
      friend std::ostream&
      operator<<(std::ostream& out, const edge_info& info) {
        out << '(' << info.separator
            << ' ' << info.property
            << ')';
        return out;
      }
    }; // struct region_graph

    // Private data members
    //==========================================================================
    
    //! An index that maps variables (nodes) to vertices this region graph.
    set_index<size_t, Domain> cluster_index_;

    //! The underlying graph
    directed_graph<size_t, vertex_info, edge_info> graph_;

    //! The next vertex id
    size_t next_vertex_;

  }; // class region_graph    

} // namespace libgm

#endif
