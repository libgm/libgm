#include "cluster_graph.hpp"

namespace libgm {

using Vertex = ClusterGraph::Vertex;
using Edge = Clustergraph::Edge;

struct ClusterGraph::Impl : Object::Impl {
  /// An index of clusters that permits fast superset/intersection queries.
  set_index<Vertex*, Domain> cluster_index;

  /// An index of separators that permits fast superset/intersection queries.
  set_index<edge_type, Domain> separator_index;

  /// The underlying undirected graph.
  std::vector<Vertex*> data_;
};

bool operator==(const Vertex& a, const Vertex& b); {
  return a.cluster_ == b.cluster_ && a.property_ == b.property_;
}

bool operator!=(const Vertex& a, const Vertex& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& out, const Vertex& v) {
  out << '(' << v.cluster_ << ' ' << v.property_ << ' ' << v.marked_ << ')';
  return out;
}


    void save(oarchive& ar) const {
      ar << domain << property;
    }

    void load(iarchive& ar) {
      ar >> domain >> property;
    }

    Shape shape(const ShapeMap& map, std::vector<size_t>& vec) const {
      return cluster_.shape(map, vec);
    }

bool ClusterGraph::empty() const {
  return impl().data.empty();
}

bool ClusterGraph::contains(Vertex* u) const {
  return u && u.graph_ == this;
}

bool ClusterGraph::contains(Vertex* u, Vertex* v) const {
  return contains(u) && contains(v) && u->neighbors_.count(v);
}

bool ClusterGraph::contains(UndirectedEdge<Vertex*> e) const {
  return contains(e.source(), e.target());
}

boost::iterator_range<argument_iterator> ClusterGraph::arguments() const {
  return impl().cluster_index.values();
}

size_t ClusterGraph::num_arguments() const {
  return impl().cluster_index.num_values();
}

size_t ClusterGraph::count(Arg x) const {
  return impl().cluster_index.count(x);
}

Object& ClusterGraph::operator[](Vertex* u) {
  return u->property_;
}

const Object& ClusterGraph::operator[](Vertex* u) const {
  return u->property_;
}

Object& ClusterGraph::operator[](const edge_descriptor& e) {
  return e->property_;
}

const Object& ClusterGraph::operator[](const edge_descriptor& e) const {
  return e->property_;
}

const Object& operator()(Vertex* u, Vertex* v) const {
  return edge(u, v, g).first.property_;
}

bool operator==(const ClusterGraph& other) const {
  return graph_ == other.graph_;
}

bool operator!=(const ClusterGraph& other) const {
  return graph_ != other.graph_;
}

bool ClusterGraph::is_connected() const {
  return test_connected(*this);
}

bool ClusterGraph::is_tree() const {
  return num_edges(*this) == num_vertices(*this) - 1 && connected();
}

bool ClusterGraph::has_running_intersection() const {
  const auto& cluster_index = impl().cluster_index;
  for (Arg x : cluster_index.values()) {
    size_t n = cluster_index.count(x);
    Vertex* v = cluster_index[x];
    size_t nreachable = test_tree(*this, v, [&](const edge_descriptor& e) {
        return e->separator().count(x) && e.target()->cluster().count(x);
      });
    if (nreachable != n) return false;
  }
  return true;
}

bool ClusterGraph::is_triangulated() const {
  return is_tree() && has_running_intersection();
}

std::ptrdiff_t ClusterGraph::tree_width() const {
  size_t max_size = 0;
  for (Vertex* v : *this) {
    max_size = std::max(max_size, v->cluster().size());
  }
  return std::ptrdiff_t(max_size) - 1;
}

vertex_descriptor ClusterGraph::find_cluster_cover(const Domain& dom) const {
  return impl().cluster_index.find_min_cover(dom);
}

edge_descriptor ClusterGraph::find_separator_cover(const Domain& dom) const {
  return impl().separator_index.find_min_cover(dom);
}

vertex_descriptor ClusterGraph::find_cluster_meets(const Domain& dom) const {
  return impl().cluster_index.find_max_intersection(dom);
}

edge_descriptor ClusterGraph::find_separator_meets(const Domain& dom) const {
  return impl().separator_index.find_max_intersection(dom);
}

void ClusterGraph::intersecting_clusters(const Domain& dom, vertex_visitor visitor) const {
  impl().cluster_index.intersecting_sets(dom, std::move(visitor));
}

void ClusterGraph::intersecting_separators(const Domain& dom, edge_visitor visitor) const {
  impl().separator_index.intersecting_sets(dom, std::move(visitor));
}

void ClusterGraph::compute_reachable(bool past_empty) {
  mpp_traversal(*this, nullptr, ReachableVisitor(this, past_empty));
}

void ClusterGraph::compute_reachable(bool past_empty, const Domain& filter) {
  std::unordered_set<Arg> set(filter.begin(), filter.end());
  mpp_traversal(*this, nullptr, ReachableVisitor(this, past_empty, &set));
}

void ClusterGraph::mark_subtree_cover(const Domain& dom, bool force_continuous) {
  if (empty()) { return; }

  // Initialize the vertices to be white.
  for (Vertex* v : *this) {
    v->marked_ = false;
  }

  // Compute the reachable variables for the set.
  compute_reachable(force_continuous, dom);

  // The edges that must be in the subtree are those such that the
  // reachable variables in both directions have a non-empty
  // symmetric difference.
  ankerl::unordered_dense::set<Arg> cover;
  for (UndirectedEdge<Vertex*> e : edges()) {
    Vertex* u = e.source();
    Vertex* v = e.target();
    const Domain& r1 = e->reachable_.forward;
    const Domain& r2 = e->reachable_.reverse;
    if (!std::includes(r1.begin(), r1.end(), r2.begin(), r2.end()) &&
        !std::includes(r2.begin(), r2.end(), r1.begin(), r1.end())) {
      e->marked_ = true;
      u->marked_ = true;
      v->marked_ = true;
      cover.insert(u->cluster_.begin(), u->cluster_.end());
      cover.insert(v->cluster_.begin(), v->cluster_.end());
    } else {
      e->marked_ = false;
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

Vertex* ClusterGraph::add_vertex(Domain cluster, Object vp) {
  Vertex* v = new Vertex(std::move(cluster), std::move(vp));
  impl().data.push_back(v);
  impl().cluster_index.insert(v, cluster);
  return v;
}

std::pair<edge_descriptor, bool>
ClusterGraph::add_edge(Vertex* u, Vertex* v, Domain separator, Object ep) {
  auto it = u->neighbors_.find(v);
  if (it != u->neighbors_.end()) {
    assert(subset(separator, u->cluster_));
    assert(subset(separator, v->cluster_));
    Edge* e = new Edge(std::move(separator), std::move(ep));
    u->neighbors_.emplace(v, e);
    v->neighbors_.emplace(u, e);
    impl().separator_index.insert(edge, separator);
    ++impl().num_edges;
    return {{u, v, e}, true};
  } else {
    return {{u, v, it->second}, false};
  }
}

std::pair<edge_descriptor, bool> ClusterGraph::add_edge(Vertex* u, Vertex* v, Object ep) {
  return add_edge(u, v, u->cluster_ & v->cluster_, std::move(ep));
}

void ClusterGraph::update_cluster(Vertex* u, const Domain& cluster) {
  if (u->cluster_ != cluster) {
    u->cluster_ = cluster;
    impl().cluster_index.erase(u);
    impl().cluster_index.insert(u, cluster);
  }
}

void ClusterGraph::update_separator(edge_descriptor e, const Domain& separator) {
  if (e->separator_ != separator) {
    e->separator_ = separator;
    impl().separator_index.erase(e);
    impl().separator_index.insert(e, separator);
  }
}

Vertex* ClusterGraph::merge(edge_descriptor e) {
  Vertex* u = e.source();
  Vertex* v = e.target();
  v_->cluster += u->cluster_;
  for (edg_descriptor in : u->in_edges()) {
    if (in.source() != v) {
      auto [e, inserted] = add_edge(in.source(), v, std::move(in->property_));
      assert(inserted);
    }
  }
  remove_vertex(u);
  return v;
}

void ClusterGraph::remove_vertex(Vertex* u) {
  for (auto [v, e] : u->neighbors_) {
    v->neighbors_.erase(u);
    delete e;
  }
  impl().data.erase(u);
  impl().cluster_index.erase(u);
}

void ClusterGraph::remove_edge(Vertex* u, Vertex* v) {
  impl().separator_index.erase(graph_.edge(u, v));
  graph_.remove_edge(u, v);
}

void ClusterGraph::remove_edges(Vertex* u) {
  for (UndirectedEdge<Vertex*> e : graph_.out_edges(u)) {
    separator_index_.erase(e);
  }
  graph_.remove_edges(u);
}

void ClusterGraph::remove_edges() {
  graph_.remove_edges();
  separator_index_.clear();
}

/// Removes all vertices and edges from the graph
void ClusterGraph::clear() {
  remove_edges();
  remove_vertices();
  // impl().cluster_index.clear();
  // impl().separator_index.clear();
}

void triangulate() {
  compute_reachable(true);
  for (Vertex* v : vertices()) {
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

void ClusterGraph::triangulated(MarkovNetwork& g, EliminationStrategy strategy) {
  clear();
  libgm::triangulate_maximal<Domain>(g, [&](Domain&& dom) {
      add_cluster(dom); }, strategy);
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

  // For each pair of overlapping cliques, add a candidate edge to the graph
  // Also, add edges between a distinguished vertex and all other vertices,
  // to ensure that the resulting junction tree is connected
  for (Vertex* u : vertices()) {
    intersecting_clusters(cluster(u), [&](Vertex* v) {
        if (u < v) { graph_.add_edge(u, v); }
      });
    if (root != u) { graph_.add_edge(root, u); }
  }

  // Compute the edges of a maximum spanning tree using Kruskal's algorithm
  std::vector<UndirectedEdge<Vertex*>> tree_edges;
  kruskal_minimum_spanning_tree(
    graph_,
    [&](UndirectedEdge<Vertex*> e) {
      return -int(cluster_index_.intersection_size(e.source(), e.target()));
    },
    std::back_inserter(tree_edges)
  );

  // Remove all edges and add back the computed edges
  graph_.remove_edges();
  for (UndirectedEdge<Vertex*> e : tree_edges) {
    add_edge(e.source(), e.target());
  }
}

/**
 * An edge visitor that computes the reachable vars in the cluster graph.
 * When using with a message passing protocol (MPP) traversal on a tree,
 * this visitor is guaranteed to compute the reachable variables for each
 * edge the tree.
 */
class ClusterGraph::ReachableVisitor {
public:
  reachable_visitor(graph_type& graph,
                    bool propagate_past_empty,
                    const std::unordered_set<Arg>* filter = nullptr)
    : graph_(graph),
      propagate_past_empty_(propagate_past_empty),
      filter_(filter) { }

  void operator()(UndirectedEdge<Vertex*> e) const {
    Domain r;
    if (!graph_[e].property.domain.empty() || propagate_past_empty_) {
      // extract the (possibly filtered) variables from the cluster
      for (Arg x : graph_[e.source()].cluster) {
        if (!filter_ || filter_->count(x)) { r.push_back(x); }
      }
      // compute the union of the incoming reachable variables
      for (UndirectedEdge<Vertex*> in : graph_.in_edges(e.source())) {
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

}; // class ReachableVisitor


void ClusterGraphBase::after_load()
{
  cluster_index_.clear();
  separator_index_.clear();
  for (vertex_type v : graph_.vertices()) {
    cluster_index_.insert(v, cluster(v));
  }
  for (UndirectedEdge<id_t> e : graph_.edges()) {
    id_t u = e.source();
    id_t v = e.target();
    graph_[e].index(u, v) = cluster(u).index(separator(e));
    graph_[e].index(v, u) = cluster(v).index(separator(e));
    separator_index_.insert(e, separator(e));
  }
}

std::ostream& operator<<(std::ostream& out, conset ClusterGraphBase& base) {
  out << g.graph_;
  return out;
}

size_t in_degree(Vertex* u, const ClusterGraph&) {
  return u->degree();
}

size_t in_degree(Vertex* u, const ClusterGraph&) {
  return u->degree();
}

size_t degree(Vertex* u, const ClusterGraph&) {
  return u->degree();
}

size_t num_vertices(const ClusterGraph& g) {
  return g.impl().data.size();
}

size_t num_edges(const ClusterGraph& g) const {
  return g.impl().num_edges;
}

std::pair<edge_descriptor, bool> ClusterGraph::edge(Vertex* u, Vertex* v, const ClusterGraph& g) {
  return {u, v, u->neighbors.at(v)};
}
