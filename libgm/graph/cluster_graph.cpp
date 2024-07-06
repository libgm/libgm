#include "cluster_graph.hpp"

#include <libgm/argument/domain_index.hpp>
#include <libgm/graph/base.hpp>
#include <libgm/graph/util/bidirectional.hpp>
#include <libgm/graph/algorithm/mst.hpp>
#include <libgm/graph/algorithm/test_connected.hpp>
#include <libgm/graph/algorithm/test_tree.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/algorithm/triangulate.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace libgm {

/**
 * The information stored with each vertex of the cluster graph.
 *
 * The base class stores the cluster along with the id of the vertex.
 * The intrusive list is used to maintain the correspo
 */
struct ClusterGraph::Vertex : VertexBase, Domain {
  /// The vertex property.
  Object property;

  /// The cluster graph owning this vertex.
  const ClusterGraph* graph;

  /// The index of the vertex (useful for vertex_index map).
  size_t index = -1;

  /// The color associated with a vertex.
  boost::default_color_type color;

  /// True if the vertex has been marked. This field is not serialized.
  bool marked = false;

  /// The outgoing edges from this node.
  OutEdgeSet out_edges;

  /// Default constructor. Default-initializes the property.
  Vertex() = default;

  Vertex(Domain cluster, Object object, const ClusterGraph* graph)
    : Domain(std::move(cluster)), property(std::move(property)), graph(graph) { }

  static Vertex* from_domain(const Domain* domain) {
    return static_cast<Vertex*>(const_cast<Domain*>(domain));
  }

  const Domain& cluster() const {
    return *this;
  }

  Domain& cluster() const {
    return *this;
  }

  void save(oarchive& ar) const {
    ar << cluster() << property << id << neighbors;
  }

  void load(iarchive& ar) {
    ar >> cluster() >> property >> id >> neighbors;
  }

  friend std::ostream& operator<<(std::ostream& out, Vertex* v) {
    out << v->id;
    return out;
  }

  friend std::ostream& operator<<(std::ostream& out, Vertex& v) {
    out << v.id << '(' << v.cluster() << ", " << v.property << ", " << v.marked << ')';
    return out;
  }

}; // struct Vertex


/**
 * The information stored with each edge of the cluster graph.
 *
 * The base class stores the cluster along with the id of the vertex.
 */
struct ClusterGraph::Edge : EdgeBase, Domain {
  /// The edge property associated with the edge.
  Object property;

  /**
   * For edge = (u, v), reachable(e) stores the variables in the subtree rooted at u,
   * away from v, in the sorted order. This field is not serialized.
   */
  Bidirectional<Domain> reachable;

  /// True if the edge has been marked. This field is not serialized.
  bool marked = false;

  Edge(Vertex* source, Vertex* target, Domain separator, Object property)
    : Domain(std::move(separator)), endpoints{source, target}, property(std::move(property)) {}

  static Edge* from_domain(const Domain* domain) {
    return static_cast<Edge*>(const_cast<Domain*>(domain));
  }

  const Domain& separator() const {
    return *this;
  }

  Domain& separator() {
    return *this;
  }

  /// Serialize members.
  void save(oarchive& ar) const {
    ar << separator() << endpoints << property;
  }

  /// Deserialize members
  void load(oiarchive& ar) {
    ar >> separator() >> endpoints >> property;
  }

  /// Outputs the edge information to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Edge& e) {
    out << '(' << e.spearator() << ' ' << e.property << ' ' << e.marked << ')';
    return out;
  }
}; // class Edge


struct ClusterGraph::Impl : Object::Impl {
  /// An index of clusters that permits fast superset/intersection queries.
  DomainIndex cluster_index;

  /// An index of separators that permits fast superset/intersection queries.
  DomainIndex separator_index;

  /// The list of all vertices.
  VertexList vertices;

  /// The list of all edges.
  EdgeList edges;
};

boost::iterator_range<ClusterGraph::out_edge_iterator> ClusterGraph::out_edges(Vertex* u) const {
  return { u->out_edges.begin(), u->out_edges.end() };
}

boost::iterator_range<ClusterGraph::in_edge_iterator> ClusterGraph::in_edges(Vertex* u) const {
  return { u->out_edges.begin(), u->out_edges.end() };
}

boost::iterator_range<ClusterGraph::adjacency_iterator> ClusterGraph::adjacent_vertices(Vertex* u)
const {
  return { u->out_edges.begin(), u->out_edges.end() };
}

boost::iterator_range<ClusterGraph::vertex_iterator> ClusterGraph::vertices() const {
  return { impl().vertices.begin(), impl().vertices.end() };
}

boost::iterator_range<ClusterGraph::edge_iterator> ClusterGraph::edges() const {
  return { impl().edges.begin(), impl().edges.end() };
}

bool ClusterGraph::empty() const {
  return impl().vertices.empty();
}

bool ClusterGraph::contains(Vertex* u) const {
  return u && u->graph == this;
}

size_t ClusterGraph::num_vertices() {
  return impl().vertices.size();
}

size_t ClusterGraph::num_edges() const {
  return impl().edges.size();
}

size_t ClusterGraph::in_degree(Vertex* u) {
  return u->out_edges.size();
}

size_t ClusterGraph::out_degree(Vertex* u) {
  return u->out_edges.size();
}

size_t ClusterGraph::degree(Vertex* u) {
  return u->out_edges.size();
}

edge_descriptor ClusterGraph::edge(Vertex* u, Vertex* v) {
  auto it = u->out_edges.find(v);
  if (it != u->out_edges.end()) {
    return *it;
  } else {
    return {};
  }
}

bool ClusterGraph::contains(Vertex* u, Vertex* v) const {
  return contains(u) && contains(v) && u->out_edges.contains(v);
}

bool ClusterGraph::contains(edge_descriptor e) const {
  return contains(e.source(), e.target());
}

Object& ClusterGraph::operator[](Vertex* u) {
  return u->property;
}

const Object& ClusterGraph::operator[](Vertex* u) const {
  return u->property;
}

Object& ClusterGraph::operator[](Edge* e) {
  return e->property;
}

const Object& ClusterGraph::operator[](Edge* e) const {
  return e->property;
}

size_t ClusterGraph::num_arguments() const {
  return impl().cluster_index.num_arguments();
}

size_t ClusterGraph::count(Arg x) const {
  return impl().cluster_index.count(x);
}

boost::iterator_range<argument_iterator> ClusterGraph::arguments() const {
  return impl().cluster_index.arguments();
}

const Domain& ClusterGraph::cluster(Vertex* v) const {
  return *v;
}

const Domain& ClusterGraph::separator(Edge* e) const {
  return *e;
}

ShapeVec ClusterGraph::shape(Vertex* v, const ShapeMap& map) const {
  return v->shape(map);
}

ShapeVec ClusterGraph::shape(Edge* e, const ShapeMap& map) const {
  return e->shape(map);
}

Dims ClusterGraph::dims(Vertex* v, const Domain& dom) const {
  return v->dims(dom);
}

Dims ClusterGraph::dims(Edge* e, const Domain& dom) const {
  return e->dims(dom);
}

Dims ClusterGraph::source_dims(edge_descriptor e) const {
  return e.source()->dims(*e);
}

Dims ClusterGraph::target_dims(edge_descriptor e) const {
  return e.target()->dims(*e);
}

MarkovNetwork<void, void> ClusterGraph::markov_network() const {
  MarkovNetwork<void, void> mn;
  for (Vertex* v : vertices()) {
    mn.make_clique(*v);
  }
  return mn;
}

bool ClusterGraph::is_connected() const {
  if (empty()) return true;

  // Visitor counting the visited vertices.
  size_t count = 0;
  struct Visitor : boost::default_bfs_visitor {
    size_t& count;

    void discover_vertex(Vertex*, const ClusterGraph&) {
      ++count;
    }
  } visitor{count};


  // Perform the BFS
  boost::breadth_first_search(*this, *vertices().begin(), queue, visitor, vertex_color_map());

  // The graph is connected if we successfully visited all vertices.
  return count == num_vertices();
}

bool ClusterGraph::is_tree() const {
  return num_edges() == num_vertices() - 1 && is_connected();
}

bool ClusterGraph::has_running_intersection() const {
  const auto& cluster_index = impl().cluster_index;

  // Initialize the color of all vertices
  reset_color();

  // Queue of vertices
  std::queue<Vertex*> queue;
  std::vector<Vertex*> examined;

  for (Arg x : cluster_index.arguments()) {
    size_t n = cluster_index.count(x);
    Vertex* v = domain_to_vertex(cluster_index[x]);

    // Visitor counting the visited vertices and filtering edges.
    size_t nreachable = 0;
    struct Visitor : boost::default_bfs_visitor {
      size_t& count;
      std::vector<Vertex*>& examined;
      Arg x;

      void discover_vertex(Vertex* v, const ClusterGraph&) {
        ++count;
        examined.push_back(v);
      }

      void examine_edge(edge_descriptor e, const ClusterGraph& g) {
        if (!g.impl().ordering.contains(x, *e) ||
            !g.impl().ordering.contains(x, *e.target())) {
          e.target()->color = boost::black_color;
          examined.push_back(e.target());
        }
      }
    } visitor{nreachable, examined, x};

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

bool ClusterGraph::is_triangulated() const {
  return is_tree() && has_running_intersection();
}

std::ptrdiff_t ClusterGraph::tree_width() const {
  size_t max_size = 0;
  for (Vertex* v : vertices()) {
    max_size = std::max(max_size, v->size());
  }
  return std::ptrdiff_t(max_size) - 1;
}

Vertex* ClusterGraph::find_cluster_cover(const Domain& dom) const {
  return Vertex::from_domain(impl().cluster_index.find_min_cover(dom, ordering));
}

Edge* ClusterGraph::find_separator_cover(const Domain& dom) const {
  return Edge::from_domain(impl().separator_index.find_min_cover(dom, ordering));
}

Vertex* ClusterGraph::find_cluster_meets(const Domain& dom) const {
  return Vertex::from_domain(impl().cluster_index.find_max_intersection(dom, ordering));
}

Edge* ClusterGraph::find_separator_meets(const Domain& dom) const {
  return Edge::from_domain(impl().separator_index.find_max_intersection(dom, ordering));
}

void ClusterGraph::intersecting_clusters(const Domain& dom, vertex_visitor visitor) const {
  for (auto handle : impl().cluster_index.find_intersections(dom)) {
    visitor(Vertex::from_domain(handle));
  }
}

void ClusterGraph::intersecting_separators(const Domain& dom, edge_visitor visitor) const {
  for (auto handle : impl().separator_index.find_intersections(dom)) {
    visitor(Edge::from_domain(handle));
  }
}

/**
 * An edge visitor that computes the reachable vars in the cluster graph.
 * When using with a message passing protocol (MPP) traversal on a tree,
 * this visitor is guaranteed to compute the reachable variables for each
 * edge the tree.
 */
class ReachableVisitor {
public:
  ReachableVisitor(const ClusterGraph* graph,
                   bool propagate_past_empty,
                   const ankerl::unordered_dense::set<Arg>* filter = nullptr)
    : graph_(graph),
      propagate_past_empty_(propagate_past_empty),
      filter_(filter) { }

  void operator()(ClusterGraph::edge_descriptor e) const {
    Domain r;
    if (!e->empty() || propagate_past_empty_) {
      // extract the (possibly filtered) variables from the cluster
      for (Arg x : *e.source()) {
        if (!filter_ || filter_->count(x)) { r.push_back(x); }
      }

      // compute the union of the incoming reachable variables
      for (auto in : graph_.in_edges(e.source())) {
        if (in.source() != e.target()) {
          r.append(in->reachable(in));
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

}; // class ReachableVisitor

void ClusterGraph::compute_reachable(bool past_empty) {
  mpp_traversal(nullptr, ReachableVisitor(this, past_empty));
}

void ClusterGraph::compute_reachable(bool past_empty, const Domain& filter) {
  ankerl::unordered_dense::set<Arg> set(filter.begin(), filter.end());
  mpp_traversal(nullptr, ReachableVisitor(this, past_empty, &set));
}

void ClusterGraph::mark_subtree_cover(const Domain& dom, bool force_continuous) {
  if (empty()) { return; }

  // Initialize the vertices to be white.
  for (Vertex* v : vertices()) {
    v->marked = false;
  }

  // Compute the reachable variables for the set.
  compute_reachable(force_continuous, dom);

  // The edges that must be in the subtree are those such that the
  // reachable variables in both directions have a non-empty
  // symmetric difference.
  ankerl::unordered_dense::set<Arg> cover;
  for (edge_descriptor e : edges()) {
    Vertex* u = e.source();
    Vertex* v = e.target();
    const Domain& r1 = e->reachable.forward;
    const Domain& r2 = e->reachable.reverse;
    if (!subset(r1, r2) && !subset(r2, r1)) {
      e->marked = true;
      u->marked = true;
      v->marked = true;
      cover.insert(u->begin(), u->end());
      cover.insert(v->begin(), v->end());
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
      uncovered.insert(uncovered.end(), x);
    }
  }
  while (!uncovered.empty()) {
    Vertex* v = find_cluster_meets(uncovered);
    assert(v);
    uncovered = uncovered - cluster(v);
    graph_[v].marked = true;
  }
}

void ClusterGraph::pre_order_traversal(Arg* start, EdgeVisitor edge_visitor) const {
  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;

    void tree_edge(edge_descriptor e, const ClusterGraph& g) {
      visit_edge(e);
    }

    void black_target(edge_descriptor e, const ClusterGraph& g) {
      throw std::invalid_argument("ClusterGraph::pre_order_traversal: detected a loop");
    }
  } visitor{std::move(edge_visitor)};

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void ClusterGraph::post_order_traversal(Arg* start, EdgeVisitor edge_visitor) const {
  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;

    void finish_edge(edge_descriptor e, const ClusterGraph& g) {
      if (e.target()->color != boost::gray_color) {
        visit_edge(e.reverse());
      }
    }

    void black_target(edge_descriptor e, const ClusterGraph& g) {
      throw std::invalid_argument("ClusterGraph::post_order_traversal: detected a loop");
    }
  } visitor{std::move(edge_visitor)};

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

void ClusterGraph::mpp_traversal(Arg* start, EdgeVisitor edge_visitor) const {
  if (empty()) return;
  if (!start) start = root();

  // Set the color of all vertices (needed because we use dfs_visit, rather than dfs_search).
  reset_color();

  // Initialize the visitor
  struct Visitor : boost::default_dfs_visitor {
    EdgeVisitor visit_edge;

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
  } visitor{std::move(edge_visitor)};

  // Run the DFS
  boost::depth_first_visit(*this, start, visitor, vertex_color_map());
}

Vertex* ClusterGraph::add_vertex(Domain cluster, Object property) {
  Vertex* v = new Vertex(std::move(cluster), std::move(property), this);
  impl().cluster_index.insert(v);
  impl().vertices.push_back(v);
  // TODO: id
  return v;
}

std::pair<edge_descriptor, bool> ClusterGraph::add_edge(Vertex* u, Vertex* v, Domain separator, Object ep) {
  assert(u != v);

  auto it = u->neighbors.find(v);
  if (it != u->neighbors.end()) {
    return {it->second, false};
  }

  assert(subset(separator, u->cluster));
  assert(subset(separator, v->cluster));
  Edge* e = new Edge(u, v, std::move(separator), std::move(ep));
  impl().separator_index.insert(e);
  u->neighbors.emplace(v, edge_descriptor::primary(e));
  v->neighbors.emplace(u, edge_descriptor::reverse(e));
  return {edge_descriptor(e), true};
}

std::pair<edge_descriptor, bool> ClusterGraph::add_edge(Vertex* u, Vertex* v, Object ep) {
  return add_edge(u, v, ordering.intersect(*u, *v), std::move(ep));
}

void ClusterGraph::update_cluster(Vertex* u, const Domain& cluster) {
  if (*u != cluster) {
    impl().cluster_index.erase(u);
    *u = cluster;
    impl().cluster_index.insert(u);
  }
}

void ClusterGraph::update_separator(Edge* e, const Domain& separator) {
  if (*e != separator) {
    impl().separator_index.erase(e);
    *e = separator;
    impl().separator_index.insert(e);
  }
}

Vertex* ClusterGraph::merge(Edge* e) {
  Vertex* u = e->endpoint[0];
  Vertex* v = e->endpoint[1];

  // Copy the edges adjacent to u
  for (auto [target, out] : u->out_edges) {
    if (target != v) {
      bool inserted = add_edge(v, target, *out, std::move(out->property)).second;
      assert(inserted);
    }
  }

  // Update the cluster at v
  update_cluster(v, ordering.union_(*v, *u));

  // Remove the dead node
  remove_vertex(u);
  return v;
}

void ClusterGraph::remove_vertex(Vertex* u) {
  clear_vertex(u);
  impl().vertices.erase(u);
  impl().cluster_index.erase(u);
  assert(false); // TODO: update the index
  delete u;
}

void ClusterGraph::remove_edge(Edge* e) {
  --impl().num_edges;
  impl().separator_index.erase(e);
  e->target->out_edges.erase(e->source);
  e->source->out_edges.erase(e->target);
  delete e;
}

void ClusterGraph::clear_vertex(Vertex* u) {
  impl().num_edges -= u->out_edges.size();
  for (auto [target, out] : u->out_edges) {
    Edge* e = out;
    separator_index.erase(e);
    target->out_edges.erase(u);
    delete e;
  }
  u->out_edges.clear();
}

void ClusterGraph::remove_edges() {
  for (Vertex* v : vertices()) {
    for (auto [_, out] : v->out_edges) {
      if (out.primary()) delete static_cast<Edge*>(out);
    }
  }
  impl().separator_index.clear();
  impl().num_edges = 0;
}

/// Removes all vertices and edges from the graph
void ClusterGraph::clear() {
  remove_edges();
  remove_vertices();
}

void ClusterGraph::reset_color() {
  for (Vertex* v : vertices()) {
    v->color = boost::white_color;
  }
}

void ClusterGraph::triangulated(MarkovNetwork& g, EliminationStrategy strategy) {
  clear();
  g.eliminate(strategy, [&](MarkovNetwork::vertex_descriptor v) {
    Domain clique(g.degree(v));
    clique[0] = v;
    boost::range::copy(g.adjacent_vertices(v), clique.begin() + 1);
    clique.sorted();
    if (impl().cluster_index.is_maximal(clique)) {
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
    intersecting_clusters(cluster(u), [this](Vertex* v) {
        if (u->id < v->id) { add_edge(u, v); }
      });
    if (root != u) { add_edge(root, u); }
  }

  // The pairs of endpoints of the MST edges, along with an output iterator populating them.
  std::vector<std::pair<Vertex*, Vertex*>> tree_edges;
  auto output = boost::make_function_output_iterator([&tree_edges](edge_descriptor e) {
    tree_edges.emplace_back(e.source(), s.target());
  });

  // The property map that returns the weight of each edge. This is the negative separator size.
  // Note this property is called multiple times for each edge, so it needs to be O(1).
  auto weight = boost::make_function_property_map<edge_descriptor>([this](edge_descriptor e) {
    return -ptrdiff_t(e->size());
  });

  // Compute the edges of a maximum spanning tree using Kruskal's algorithm.
  boost::kruskal_minimum_spanning_tree(*this, output, boost::weight_map(weight));

  // Remove all edges and add back the MST edges.
  remove_edges();
  for (auto [s, t] : tree_edges) {
    add_edge(s, t);
  }
}

// void ClusterGraphBase::after_load()
// {
//   cluster_index.clear();
//   separator_index.clear();
//   for (vertex_type v : graph_.vertices()) {
//     cluster_index.insert(v, cluster(v));
//   }
//   for (UndirectedEdge<id_t> e : graph_.edges()) {
//     id_t u = e.source();
//     id_t v = e.target();
//     graph_[e].index(u, v) = cluster(u).index(separator(e));
//     graph_[e].index(v, u) = cluster(v).index(separator(e));
//     separator_index.insert(e, separator(e));
//   }
// }

std::ostream& operator<<(std::ostream& out, conset ClusterGraphBase& base) {
  out << g.graph_;
  return out;
}

boost::default_color_type get(const ClusterGraph::ColorMap&, ClusterGraph::Vertex* v) {
  return v->color;
}

/// Sets the color associated with a vertex.
void put(const ClusterGraph::ColorMap&, ClusterGraph::Vertex* v, boost::default_color_type c) {
  v->color = c;
}

} // namespace libgm