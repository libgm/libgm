#include "cluster_graph.hpp"

#include <libgm/datastructure/domain_index_operations.hpp>

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/range/algorithm.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace libgm {

/**
 * The information stored with each vertex of the cluster graph.
 *
 * The base class stores the cluster along with the id of the vertex.
 * The intrusive list is used to maintain the correspo
 */
struct ClusterGraph::Vertex {
  /// The cluster associated with the vertex.
  Domain cluster;

  /// The vertex property.
  Object property;

  /// The cluster graph owning this vertex.
  const Impl* impl;

  /// The index of the vertex (useful for vertex_index map).
  size_t index = -1;

  /// The color associated with a vertex.
  boost::default_color_type color;

  /// True if the vertex has been marked. This field is not serialized.
  bool marked = false;

  /// The degree of the vertex.
  size_t degree = 0;

  /// The adjacent edges of this vertex.
  IntrusiveList<Edge> adjacency;

  /// The hook for intrusive list of all vertices.
  IntrusiveList<Vertex>::Hook hook;

  /// The hooks for intrusive lists within cluster index.
  IntrusiveList<Vertex>::HookArray index_hooks;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(CEREAL_NVP(cluster), CEREAL_NVP(property));
    if constexpr (ARCHIVE::is_loading::value) {
      index_hooks.reset(cluster.size());
    }
  }

  Vertex(Impl* impl)
    : impl(impl) {}

  Vertex(Domain cluster, Object property, Impl* impl)
    : cluster(std::move(cluster)),
      property(std::move(property)),
      impl(impl),
      index_hooks(domain().size()) { }

  const Domain& domain() const {
    return cluster;
  }

  friend std::ostream& operator<<(std::ostream& out, Vertex& v) {
    out << v.index << '(' << v.cluster << ", " << v.property << ", " << v.marked << ')';
    return out;
  }
}; // struct Vertex

/**
 * The information stored with each edge of the cluster graph.
 *
 * The base class stores the cluster along with the id of the vertex.
 */
struct ClusterGraph::Edge {
  /// The connectivity
  IntrusiveEdge<Vertex, Edge>::Connectivity connectivity;

  /// The separator assocociated with this edge.
  Domain separator;

  /// The edge property associated with the edge.
  Object property;

  /// The implementation that this edge belongs to.
  Impl* impl;

  /**
   * For edge = (u, v), reachable(e) stores the variables in the subtree rooted at u,
   * away from v, in the sorted order. This field is not serialized.
   */
  Domain reachable[2];

  /// True if the edge has been marked. This field is not serialized.
  bool marked = false;

  /// The hook for an intrusive list of all edges.
  IntrusiveList<Edge>::Hook hook;

  /// The hooks for adjacency lists.
  IntrusiveList<Edge>::Hook adjacency_hook[2];

  /// The hook for an intrusive lists within separator index.
  IntrusiveList<Edge>::HookArray index_hooks;

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) {
    ar(cereal::make_nvp("u", u()->index));
    ar(cereal::make_nvp("v", v()->index));
    ar(CEREAL_NVP(separator), CEREAL_NVP(property));
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    // FIXME
    // ar(cereal::make_nvp("u", u.index));
    // ar(cereal::make_nvp("v", v.index));
    ar(CEREAL_NVP(separator), CEREAL_NVP(property));
    index_hooks.reset(separator.size());
  }

  Edge(Impl* impl) : impl(impl) {}

  Edge(Vertex* u, Vertex* v, Domain separator, Object property, Impl* impl)
    : connectivity{u, v},
      separator(std::move(separator)),
      property(std::move(property)),
      impl(impl),
      index_hooks(domain().size()) {}

  ~Edge() {
    --u()->degree;
    --v()->degree;
  }

  const Domain& domain() const {
    return separator;
  }

  Vertex* u() const {
    return connectivity.vertex[0];
  }

  Vertex* v() const {
    return connectivity.vertex[1];
  }

  /// Outputs the edge information to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Edge& e) {
    out << '(' << e.separator << ' ' << e.property << ' ' << e.marked << ')';
    return out;
  }
}; // class Edge

static_assert(std::is_standard_layout_v<ClusterGraph::Vertex>);
static_assert(std::is_standard_layout_v<ClusterGraph::Edge>);

struct ClusterGraph::Impl : Object::Impl {
  /// The list of all vertices.
  IntrusiveList<Vertex> vertices;

  /// The list of all edges.
  IntrusiveList<Edge> edges;

  /// The total number of vertices.
  size_t num_vertices = 0;

  /// The total number of edges.
  size_t num_edges = 0;

  /// An index of clusters that permits fast superset/intersection queries.
  DomainIndex<Vertex> cluster_index;

  /// An index of separators that permits fast superset/intersection queries.
  DomainIndex<Edge> separator_index;

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) const {
    size_t i = 0;
    ar(cereal::make_size_tag(num_vertices));
    for (Vertex* vertex : vertices) {
      ar(*vertex);
      vertex->index = i++;
    }

    ar(cereal::make_size_tag(num_edges));
    for (Edge* edge : edges) {
      ar(*edge);
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    cereal::size_type size;

    // Deserialize the vertices
    ar(cereal::make_size_tag(size));
    num_vertices = size;
    for (size_t i = 0; i < num_vertices; ++i) {
      Vertex* vertex = new Vertex(this);
      ar(*vertex);
      vertices.push_back(vertex, vertex->hook);
      cluster_index.insert(vertex, vertex->index_hooks);
    }

    // Deserialize the edges
    std::vector<Vertex*> indexed_vertices(vertices.begin(), vertices.end());
    ar(cereal::make_size_tag(size));
    num_edges = size;
    for (size_t i = 0; i < num_edges; ++i) {
      // Load the edge
      Edge* edge = new Edge(this);
      ar(*edge);

      // Add the edge to the edge and adjacency lists
      edges.push_back(edge, edge->hook);
      edge->u()->adjacency.push_back(edge, edge->connectivity.adjacency_hook[0]);
      edge->v()->adjacency.push_back(edge, edge->connectivity.adjacency_hook[1]);
      ++edge->u()->degree;
      ++edge->v()->degree;

      // Add the edge to the separator index
      separator_index.insert(edge, edge->index_hooks);
    }
  }

  Impl() = default;
};

SubRange<ClusterGraph::out_edge_iterator> ClusterGraph::out_edges(Vertex* u) const {
  return u->adjacency.entries();
}

SubRange<ClusterGraph::in_edge_iterator> ClusterGraph::in_edges(Vertex* u) const {
  return out_edges(u);
}

SubRange<ClusterGraph::adjacency_iterator> ClusterGraph::adjacent_vertices(Vertex* u) const {
  return out_edges(u);
}

SubRange<ClusterGraph::vertex_iterator> ClusterGraph::vertices() const {
  return { impl().vertices.begin(), impl().vertices.end() };
}

SubRange<ClusterGraph::edge_iterator> ClusterGraph::edges() const {
  return { impl().edges.begin(), impl().edges.end() };
}

bool ClusterGraph::empty() const {
  return impl().vertices.empty();
}

bool ClusterGraph::contains(Vertex* u) const {
  return u && u->impl == &impl();
}

size_t ClusterGraph::num_vertices() const {
  return impl().num_vertices;
}

size_t ClusterGraph::num_edges() const {
  return impl().num_edges;
}

size_t ClusterGraph::in_degree(Vertex* u) const {
  return u->degree;
}

size_t ClusterGraph::out_degree(Vertex* u) const {
  return u->degree;
}

size_t ClusterGraph::degree(Vertex* u) const {
  return u->degree;
}

bool ClusterGraph::contains(edge_descriptor e) const {
  return e->impl == &impl();
}

Object& ClusterGraph::operator[](Vertex* u) {
  return u->property;
}

const Object& ClusterGraph::operator[](Vertex* u) const {
  return u->property;
}

Object& ClusterGraph::operator[](edge_descriptor e) {
  return e->property;
}

const Object& ClusterGraph::operator[](edge_descriptor e) const {
  return e->property;
}

SubRange<ClusterGraph::argument_iterator> ClusterGraph::arguments() const {
  return impl().cluster_index.arguments();
}

const Domain& ClusterGraph::cluster(Vertex* v) const {
  return v->cluster;
}

const Domain& ClusterGraph::separator(edge_descriptor e) const {
  return e->separator;
}

Shape ClusterGraph::shape(Vertex* v, const ShapeMap& map) const {
  return v->cluster.shape(map);
}

Shape ClusterGraph::shape(edge_descriptor e, const ShapeMap& map) const {
  return e->separator.shape(map);
}

Dims ClusterGraph::dims(Vertex* v, const Domain& dom) const {
  return v->cluster.dims(dom);
}

Dims ClusterGraph::dims(edge_descriptor e, const Domain& dom) const {
  return e->separator.dims(dom);
}

Dims ClusterGraph::source_dims(edge_descriptor e) const {
  return e.source()->cluster.dims(e->separator);
}

Dims ClusterGraph::target_dims(edge_descriptor e) const {
  return e.target()->cluster.dims(e->separator);
}

MarkovNetworkT<> ClusterGraph::markov_network() const {
  MarkovNetworkT<> mn;
  for (Vertex* v : vertices()) {
    mn.add_clique(v->cluster);
  }
  return mn;
}

bool ClusterGraph::is_connected() {
  if (empty()) return true;

  // Visitor counting the visited vertices.
  size_t count = 0;
  struct Visitor : boost::default_bfs_visitor {
    size_t& count;
    Visitor(size_t& count) : count(count) {}

    void discover_vertex(Vertex*, const ClusterGraph&) {
      ++count;
    }
  } visitor(count);


  // Perform the BFS
  boost::queue<Vertex*> queue;
  boost::breadth_first_search(*this, *vertices().begin(), queue, visitor, vertex_color_map());

  // The graph is connected if we successfully visited all vertices.
  return count == num_vertices();
}

bool ClusterGraph::is_tree() {
  return num_edges() == num_vertices() - 1 && is_connected();
}

bool ClusterGraph::has_running_intersection() {
  const auto& cluster_index = impl().cluster_index;

  // Initialize the color of all vertices
  reset_color();

  // Queue of vertices
  boost::queue<Vertex*> queue;
  std::vector<Vertex*> examined;

  for (Arg x : cluster_index.arguments()) {
    size_t n = cluster_index.count(x);
    Vertex* v = cluster_index[x];

    // Visitor counting the visited vertices and filtering edges.
    size_t nreachable = 0;
    struct Visitor : boost::default_bfs_visitor {
      size_t& count;
      std::vector<Vertex*>& examined;
      Arg x;

      Visitor(size_t& count, std::vector<Vertex*>& examined, Arg x)
        : count(count), examined(examined), x(x) {}

      void discover_vertex(Vertex* v, const ClusterGraph&) {
        ++count;
        examined.push_back(v);
      }

      void examine_edge(edge_descriptor e, const ClusterGraph& g) {
        if (!e->separator.contains(x) || !e.target()->cluster.contains(x)) {
          e.target()->color = boost::black_color;
          examined.push_back(e.target());
        }
      }
    } visitor(nreachable, examined, x);

    // Perform the BFS
    examined.clear();
    boost::breadth_first_visit(*this, v, queue, visitor, vertex_color_map());
    if (nreachable != n) return false;

    // Revert the color of all examined vertices
    for (Vertex* v : examined) {
      v->color = boost::white_color;
    }
  }

  return true;
}

bool ClusterGraph::is_triangulated() {
  return is_tree() && has_running_intersection();
}

int ClusterGraph::tree_width() const {
  int max_size = 0;
  for (Vertex* v : vertices()) {
    max_size = std::max(max_size, int(v->cluster.size()));
  }
  return max_size - 1;
}

ClusterGraph::Vertex* ClusterGraph::find_cluster_cover(const Domain& dom) const {
  return find_min_cover(impl().cluster_index, dom);
}

ClusterGraph::edge_descriptor ClusterGraph::find_separator_cover(const Domain& dom) const {
  return find_min_cover(impl().separator_index, dom);
}

ClusterGraph::Vertex* ClusterGraph::find_cluster_meets(const Domain& dom) const {
  return find_max_intersection(impl().cluster_index, dom);
}

ClusterGraph::edge_descriptor ClusterGraph::find_separator_meets(const Domain& dom) const {
  return find_max_intersection(impl().separator_index, dom);
}

void ClusterGraph::intersecting_clusters(const Domain& dom, VertexVisitor visitor) const {
  visit_intersections(impl().cluster_index, dom, std::move(visitor));
}

void ClusterGraph::intersecting_separators(const Domain& dom, EdgeVisitor visitor) const {
  visit_intersections(impl().separator_index, dom, std::move(visitor));
}

/**
 * An edge visitor that computes the reachable vars in the cluster graph.
 * When using with a message passing protocol (MPP) traversal on a tree,
 * this visitor is guaranteed to compute the reachable variables for each
 * edge the tree.
 */
class ReachableVisitor {
public:
  ReachableVisitor(bool propagate_past_empty, ArgSet* filter = nullptr)
    : propagate_past_empty_(propagate_past_empty),
      filter_(filter) { }

  void operator()(ClusterGraph::edge_descriptor e) const {
    Domain r;
    if (!e->separator.empty() || propagate_past_empty_) {
      // extract the (possibly filtered) variables from the cluster
      for (Arg x : e.source()->cluster) {
        if (!filter_ || filter_->count(x)) { r.push_back(x); }
      }

      // compute the union of the incoming reachable variables
      for (ClusterGraph::edge_descriptor out : e.source()->adjacency.entries()) {
        if (out.target() != e.target()) {
          r.append(out->reachable[!out.index()]);
        }
      }
      // eliminate duplicates
      r.unique();
    }

    // store the result
    e->reachable[e.index()] = r;
  }

private:
  bool propagate_past_empty_;
  const ArgSet* filter_;

}; // class ReachableVisitor

void ClusterGraph::compute_reachable(bool past_empty) {
  mpp_traversal(nullptr, ReachableVisitor(past_empty));
}

void ClusterGraph::compute_reachable(bool past_empty, const Domain& filter) {
  ankerl::unordered_dense::set<Arg> set(filter.begin(), filter.end());
  mpp_traversal(nullptr, ReachableVisitor(past_empty, &set));
}

void ClusterGraph::mark_subtree_cover(const Domain& domain, bool force_continuous) {
  if (empty()) { return; }

  // Initialize the vertices to be white.
  for (Vertex* v : vertices()) {
    v->marked = false;
  }

  // Compute the reachable variables for the set.
  compute_reachable(force_continuous, domain);

  // The edges that must be in the subtree are those such that the
  // reachable variables in both directions have a non-empty
  // symmetric difference.
  ArgSet cover;
  for (edge_descriptor e : edges()) {
    Vertex* u = e->u();
    Vertex* v = e->v();
    const Domain& r1 = e->reachable[0];
    const Domain& r2 = e->reachable[1];
    if (!is_subset(r1, r2) && !is_subset(r2, r1)) {
      e->marked = true;
      u->marked = true;
      v->marked = true;
      cover.insert(u->cluster.begin(), u->cluster.end());
      cover.insert(v->cluster.begin(), v->cluster.end());
    } else {
      e->marked = false;
    }
  }

  // We must also mark vertices that are part of the subtree but
  // are not attached to any other vertex in the subtree.
  // If force_continuous = true, then either all nodes were covered
  // in the previous stage, or the nodes are contained in a single clique
  Domain uncovered;
  for (Arg x : domain) {
    if (!cover.count(x)) {
      uncovered.push_back(x);
    }
  }
  while (!uncovered.empty()) {
    Vertex* v = find_cluster_meets(uncovered);
    assert(v);
    uncovered -= v->cluster;
    v->marked = true;
  }
}

void ClusterGraph::pre_order_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;
    Visitor(EdgeVisitor visit_edge) : visit_edge(std::move(visit_edge)) {}

    void tree_edge(edge_descriptor e, const ClusterGraph& g) {
      visit_edge(e);
    }

    void black_target(edge_descriptor e, const ClusterGraph& g) {
      throw std::invalid_argument("ClusterGraph::pre_order_traversal: detected a loop");
    }
  } visitor(std::move(edge_visitor));

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void ClusterGraph::post_order_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;
    Visitor(EdgeVisitor visit_edge) : visit_edge(std::move(visit_edge)) {}

    void finish_edge(edge_descriptor e, const ClusterGraph& g) {
      if (e.target()->color != boost::gray_color) {
        visit_edge(e.reverse());
      }
    }

    void black_target(edge_descriptor e, const ClusterGraph& g) {
      throw std::invalid_argument("ClusterGraph::post_order_traversal: detected a loop");
    }
  } visitor(std::move(edge_visitor));

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void ClusterGraph::mpp_traversal(Vertex* start, EdgeVisitor edge_visitor) {
  if (empty()) return;
  if (!start) start = root();

  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;
    Visitor(EdgeVisitor visit_edge) : visit_edge(std::move(visit_edge)) {}

    void tree_edge(edge_descriptor e, const ClusterGraph& g) {
      visit_edge(e);
    }

    void finish_edge(edge_descriptor e, const ClusterGraph& g) {
      if (e.target()->color != boost::gray_color) {
        visit_edge(e.reverse());
      }
    }

    void black_target(edge_descriptor e, const ClusterGraph& g) {
      throw std::invalid_argument("ClusterGraph::post_order_traversal: detected a loop");
    }
  } visitor(std::move(edge_visitor));

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

ClusterGraph::Vertex* ClusterGraph::add_vertex(Domain cluster, Object property) {
  Vertex* v = new Vertex(std::move(cluster), std::move(property), &impl());
  impl().vertices.push_back(v, v->hook);
  impl().cluster_index.insert(v, v->index_hooks);
  return v;
}

ClusterGraph::edge_descriptor
ClusterGraph::add_edge(Vertex* u, Vertex* v, Domain separator, Object ep) {
  assert(u != v);
  assert(is_subset(separator, u->cluster));
  assert(is_subset(separator, v->cluster));
  Edge* e = new Edge(u, v, std::move(separator), std::move(ep), &impl());
  impl().edges.push_back(e, e->hook);
  impl().separator_index.insert(e, e->index_hooks);
  u->adjacency.push_back(e, e->connectivity.adjacency_hook[0]);
  v->adjacency.push_back(e, e->connectivity.adjacency_hook[1]);
  return ClusterGraph::edge_descriptor({e, e->adjacency_hook});
}

ClusterGraph::edge_descriptor ClusterGraph::add_edge(Vertex* u, Vertex* v, Object ep) {
  return add_edge(u, v, u->cluster & v->cluster, std::move(ep));
}

void ClusterGraph::update_cluster(Vertex* u, const Domain& cluster) {
  if (u->cluster != cluster) {
    u->cluster = cluster;
    u->index_hooks.reset(cluster.size());
    impl().cluster_index.insert(u, u->index_hooks);
  }
}

void ClusterGraph::update_separator(edge_descriptor e, const Domain& separator) {
  if (e->separator != separator) {
    e->separator = separator;
    e->index_hooks.reset(separator.size());
    impl().separator_index.insert(e.get(), e->index_hooks);
  }
}

ClusterGraph::Vertex* ClusterGraph::merge(edge_descriptor e) {
  Vertex* u = e->u();
  Vertex* v = e->v();

  // Copy the edges adjacent to u
  for (edge_descriptor out : u->adjacency) {
    if (out.target() != v) {
      add_edge(v, out.target(), std::move(out->separator), std::move(out->property));
    }
  }

  // Update the cluster at v
  update_cluster(v, u->cluster | v->cluster);

  // Remove the dead node
  remove_vertex(u);
  return v;
}

void ClusterGraph::remove_vertex(Vertex* u) {
  --impl().num_vertices;
  clear_vertex(u);
  delete u;
}

void ClusterGraph::remove_edge(edge_descriptor e) {
  --impl().num_edges;
  delete e.get();
}

void ClusterGraph::clear_vertex(Vertex* u) {
  impl().num_edges -= u->degree;
  for (auto it = u->adjacency.begin(); it != u->adjacency.end();) {
    delete *it++;
  }
}

void ClusterGraph::remove_edges() {
  impl().num_edges = 0;
  for (auto it = impl().edges.begin(); it != impl().edges.end();) {
    delete *it++;
  }
}

void ClusterGraph::clear() {
  remove_edges();

  impl().num_vertices = 0;
  for (auto it = impl().vertices.begin(); it != impl().vertices.end();) {
    delete *it++;
  }
}

void ClusterGraph::reset_color() {
  for (Vertex* v : vertices()) {
    v->color = boost::white_color;
  }
}

void ClusterGraph::triangulated(MarkovNetwork& g, const EliminationStrategy& strategy) {
  clear();
  g.eliminate(strategy, [&](MarkovNetwork::vertex_descriptor v) {
    Domain clique(g.adjacent_vertices(v));
    clique.push_back(v);
    clique.sort();
    if (is_maximal(impl().cluster_index, clique)) {
      add_vertex(std::move(clique));
    };
  });
  mst_edges();
}

void ClusterGraph::triangulated(const std::vector<Domain>& cliques) {
  clear();
  for (const Domain& clique : cliques) {
    add_vertex(clique);
  }
  mst_edges();
}

void ClusterGraph::mst_edges() {
  remove_edges();
  if (empty()) { return; }

  // Select a distinguished vertex of the tree.
  Vertex* root = *vertices().begin();

  // For each pair of overlapping cliques, add a candidate edge to the graph.
  // Also, add edges between a distinguished vertex and all other vertices,
  // to ensure that the resulting junction tree is connected.
  for (Vertex* u : vertices()) {
    intersecting_clusters(cluster(u), [this, u](Vertex* v) {
      if (u < v) { add_edge(u, v); }
    });
    if (root != u) { add_edge(root, u); }
  }

  // The pairs of endpoints of the MST edges, along with an output iterator populating them.
  std::vector<std::pair<Vertex*, Vertex*>> tree_edges;

  // The property map that returns the weight of each edge. This is the negative separator size.
  // Note this property is called multiple times for each edge, so it needs to be O(1).
  auto weight = boost::make_function_property_map<edge_descriptor>([this](edge_descriptor e) {
    return -ptrdiff_t(e->separator.size());
  });

  // Compute the edges of a maximum spanning tree using Kruskal's algorithm.
  boost::kruskal_minimum_spanning_tree(
    *this, std::back_inserter(tree_edges),
    boost::weight_map(weight).vertex_index_map(vertex_index_map()));

  // Remove all edges and add back the MST edges.
  remove_edges();
  for (auto [s, t] : tree_edges) {
    add_edge(s, t);
  }
}

boost::default_color_type get(const ClusterGraph::VertexColorMap&, ClusterGraph::Vertex* v) {
  return v->color;
}

/// Sets the color associated with a vertex.
void put(const ClusterGraph::VertexColorMap&, ClusterGraph::Vertex* v, boost::default_color_type c) {
  v->color = c;
}

} // namespace libgm
