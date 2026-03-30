#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/domain_index.hpp>
#include <libgm/datastructure/domain_index_operations.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/graph/algorithm/elimination_strategies.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/undirected_graph.hpp>

#include <ankerl/unordered_dense.h>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>

#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

template <typename Item>
struct IndexedDomain {
  /// The owning graph item.
  Item* owner = nullptr;

  /// The stored domain.
  Domain args;

  /// Hooks used by the domain index.
  typename IntrusiveList<IndexedDomain<Item>>::HookArray hooks;

  IndexedDomain() = default;

  explicit IndexedDomain(Domain args)
    : args(std::move(args)),
      hooks(this->args.size()) {}

  IndexedDomain(const IndexedDomain& other)
    : owner(other.owner),
      args(other.args),
      hooks(args.size()) {}

  IndexedDomain& operator=(const IndexedDomain& other) {
    if (this != &other) {
      owner = other.owner;
      args = other.args;
      hooks.reset(args.size());
    }
    return *this;
  }

  IndexedDomain(IndexedDomain&&) noexcept = default;
  IndexedDomain& operator=(IndexedDomain&&) noexcept = default;

  const Domain& domain() const {
    return args;
  }

  void reset(Domain new_args) {
    args = std::move(new_args);
    hooks.reset(args.size());
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("domain", args));
    if constexpr (Archive::is_loading::value) {
      hooks.reset(args.size());
    }
  }
};

template <typename VP = void, typename EP = VP>
class ClusterGraph : private UndirectedGraph {
  using VertexDomain = IndexedDomain<UndirectedGraph::Vertex>;
  using EdgeDomain = IndexedDomain<UndirectedGraph::Edge>;
  using VertexAnnotation = Annotated<VertexDomain, VP>;
  using EdgeAnnotation = Annotated<EdgeDomain, EP>;
  using ReachableMap = ankerl::unordered_dense::map<UndirectedGraph::Edge*, std::array<Domain, 2>>;

public:
  /// The underlying graph vertex type.
  using UndirectedGraph::Vertex;

  /// The underlying graph edge type.
  using UndirectedGraph::Edge;
  using UndirectedGraph::VertexColorMap;
  using UndirectedGraph::VertexIndexMap;
  using UndirectedGraph::adjacent_vertices;
  using UndirectedGraph::adjacency_iterator;
  using UndirectedGraph::contains;
  using UndirectedGraph::degree_size_type;
  using UndirectedGraph::directed_category;
  using UndirectedGraph::edge_parallel_category;
  using UndirectedGraph::degree;
  using UndirectedGraph::edge_descriptor;
  using UndirectedGraph::edge_iterator;
  using UndirectedGraph::edges_size_type;
  using UndirectedGraph::edges;
  using UndirectedGraph::empty;
  using UndirectedGraph::in_degree;
  using UndirectedGraph::in_edge_iterator;
  using UndirectedGraph::in_edges;
  using UndirectedGraph::is_connected;
  using UndirectedGraph::is_tree;
  using UndirectedGraph::marked;
  using UndirectedGraph::mpp_traversal;
  using UndirectedGraph::null_vertex;
  using UndirectedGraph::num_edges;
  using UndirectedGraph::num_vertices;
  using UndirectedGraph::out_degree;
  using UndirectedGraph::out_edge_iterator;
  using UndirectedGraph::out_edges;
  using UndirectedGraph::post_order_traversal;
  using UndirectedGraph::pre_order_traversal;
  using UndirectedGraph::property;
  using UndirectedGraph::root;
  using UndirectedGraph::traversal_category;
  using UndirectedGraph::vertex_color_map;
  using UndirectedGraph::vertex_descriptor;
  using UndirectedGraph::vertex_index_map;
  using UndirectedGraph::vertex_iterator;
  using UndirectedGraph::vertices_size_type;
  using UndirectedGraph::vertices;
  /// Iterator over arguments present in the cluster index.
  using argument_iterator = DomainIndex<VertexDomain>::argument_iterator;

  /// Visitor over cluster vertices.
  using VertexVisitor = std::function<void(vertex_descriptor)>;

  /// Visitor over separator edges.
  using ModelEdgeVisitor = std::function<void(edge_descriptor)>;

  /// Reference to the typed vertex property.
  using vertex_property_reference = std::add_lvalue_reference_t<VP>;

  /// Const reference to the typed vertex property.
  using const_vertex_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;

  /// Reference to the typed edge property.
  using edge_property_reference = std::add_lvalue_reference_t<EP>;

  /// Const reference to the typed edge property.
  using const_edge_property_reference = std::add_lvalue_reference_t<std::add_const_t<EP>>;

  ClusterGraph()
    : UndirectedGraph(property_layout<VertexAnnotation>(), property_layout<EdgeAnnotation>()) {}

  ClusterGraph(const ClusterGraph& other)
    : UndirectedGraph(other) {
    rebuild_indices();
    copy_reachable(other);
  }

  ClusterGraph(ClusterGraph&& other) noexcept = default;

  ClusterGraph& operator=(const ClusterGraph& other) {
    if (this != &other) {
      UndirectedGraph::operator=(other);
      rebuild_indices();
      copy_reachable(other);
    }
    return *this;
  }

  ClusterGraph& operator=(ClusterGraph&& other) noexcept = default;

  // Domain accessors
  //--------------------------------------------------------------------------
  /// Returns the range of arguments present in any cluster.
  std::ranges::subrange<argument_iterator> arguments() const {
    return cluster_index_.arguments();
  }

  /// Returns the number of distinct arguments present in the graph.
  size_t num_arguments() const {
    return cluster_index_.num_arguments();
  }

  /// Returns the number of clusters containing the supplied argument.
  size_t count(Arg x) const {
    return cluster_index_.count(x);
  }

  /// Returns the cluster domain associated with a vertex.
  const Domain& cluster(Vertex* v) const {
    return vertex_annotation(v).value.domain();
  }

  /// Returns the separator domain associated with an edge.
  const Domain& separator(edge_descriptor e) const {
    return edge_annotation(e).value.domain();
  }

  /// Returns the cached reachable domain in the direction of `e`.
  const Domain& reachable(edge_descriptor e) const {
    return reachable_.at(e.get())[e.index()];
  }

  /// Returns the shape of the cluster at a vertex.
  Shape shape(Vertex* v, const ShapeMap& map) const {
    return cluster(v).shape(map);
  }

  /// Returns the shape of the separator at an edge.
  Shape shape(edge_descriptor e, const ShapeMap& map) const {
    return separator(e).shape(map);
  }

  /// Returns the index mapping from `dom` into the cluster at `v`.
  Dims dims(Vertex* v, const Domain& dom) const {
    return cluster(v).dims(dom);
  }

  /// Returns the index mapping from `dom` into the separator at `e`.
  Dims dims(edge_descriptor e, const Domain& dom) const {
    return separator(e).dims(dom);
  }

  /// Returns the index mapping from the separator to the source cluster.
  Dims source_dims(edge_descriptor e) const {
    return cluster(e.source()).dims(separator(e));
  }

  /// Returns the index mapping from the separator to the target cluster.
  Dims target_dims(edge_descriptor e) const {
    return cluster(e.target()).dims(separator(e));
  }

  /// Returns the typed property associated with a vertex.
  vertex_property_reference operator[](Vertex* v) {
    return vertex_annotation(v).property();
  }

  /// Returns the typed property associated with a vertex.
  const_vertex_property_reference operator[](Vertex* v) const {
    return vertex_annotation(v).property();
  }

  /// Returns the typed property associated with an edge.
  edge_property_reference operator[](edge_descriptor e) {
    return edge_annotation(e).property();
  }

  /// Returns the typed property associated with an edge.
  const_edge_property_reference operator[](edge_descriptor e) const {
    return edge_annotation(e).property();
  }

  // Queries
  //--------------------------------------------------------------------------
  /// Computes the Markov network induced by the clusters.
  MarkovNetwork markov_network() const {
    MarkovNetwork mn;
    for (Vertex* v : vertices()) {
      mn.add_clique(cluster(v));
    }
    return mn;
  }

  /// Returns true if the graph satisfies the running intersection property.
  bool has_running_intersection() {
    reset_color();

    boost::queue<Vertex*> queue;
    std::vector<Vertex*> examined;

    for (Arg x : cluster_index_.arguments()) {
      size_t n = cluster_index_.count(x);
      Vertex* v = cluster_index_[x]->owner;

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
          if (!g.separator(e).contains(x) || !g.cluster(e.target()).contains(x)) {
            put(VertexColorMap{}, e.target(), boost::black_color);
            examined.push_back(e.target());
          }
        }
      } visitor(nreachable, examined, x);

      examined.clear();
      boost::breadth_first_visit(*this, v, queue, visitor, vertex_color_map());
      if (nreachable != n) {
        return false;
      }

      for (Vertex* examined_vertex : examined) {
        put(vertex_color_map(), examined_vertex, boost::white_color);
      }
    }

    return true;
  }

  /// Returns true if the graph is a triangulated junction tree.
  bool is_triangulated() {
    return is_tree() && has_running_intersection();
  }

  /// Returns the maximum cluster size minus one.
  int tree_width() const {
    int max_size = 0;
    for (Vertex* v : vertices()) {
      max_size = std::max(max_size, int(cluster(v).size()));
    }
    return max_size - 1;
  }

  /// Returns a cluster whose domain covers `dom`, or the null vertex.
  Vertex* find_cluster_cover(const Domain& dom) const {
    VertexDomain* domain = find_min_cover(cluster_index_, dom);
    return domain ? domain->owner : nullptr;
  }

  /// Returns a separator whose domain covers `dom`, or the null edge.
  edge_descriptor find_separator_cover(const Domain& dom) const {
    EdgeDomain* domain = find_min_cover(separator_index_, dom);
    return domain ? edge_descriptor(domain->owner) : edge_descriptor();
  }

  /// Returns a cluster whose domain best intersects `dom`.
  Vertex* find_cluster_meets(const Domain& dom) const {
    VertexDomain* domain = find_max_intersection(cluster_index_, dom);
    return domain ? domain->owner : nullptr;
  }

  /// Returns a separator whose domain best intersects `dom`.
  edge_descriptor find_separator_meets(const Domain& dom) const {
    EdgeDomain* domain = find_max_intersection(separator_index_, dom);
    return domain ? edge_descriptor(domain->owner) : edge_descriptor();
  }

  /// Visits all clusters intersecting `dom`.
  void intersecting_clusters(const Domain& dom, VertexVisitor visitor) const {
    visit_intersections(cluster_index_, dom, [&](VertexDomain* domain) {
      visitor(domain->owner);
    });
  }

  /// Visits all separators intersecting `dom`.
  void intersecting_separators(const Domain& dom, ModelEdgeVisitor visitor) const {
    visit_intersections(separator_index_, dom, [&](EdgeDomain* domain) {
      visitor(edge_descriptor(domain->owner));
    });
  }

  /// Computes reachable domains on a tree, optionally stopping at empty separators.
  void compute_reachable(bool past_empty) {
    mpp_traversal(root(), ReachableVisitor(*this, past_empty));
  }

  /// Computes reachable domains and intersects them with `filter`.
  void compute_reachable(bool past_empty, const Domain& filter) {
    ankerl::unordered_dense::set<Arg> set(filter.begin(), filter.end());
    mpp_traversal(root(), ReachableVisitor(*this, past_empty, &set));
  }

  /// Marks the smallest subtree or subforest covering `domain`.
  void mark_subtree_cover(const Domain& domain, bool force_continuous) {
    if (empty()) {
      return;
    }

    for (Vertex* v : vertices()) {
      set_marked(v, false);
    }
    for (edge_descriptor e : edges()) {
      set_marked(e, false);
    }

    compute_reachable(force_continuous, domain);

    ankerl::unordered_dense::set<Arg> cover;
    for (edge_descriptor e : edges()) {
      Vertex* u = e.source();
      Vertex* v = e.target();
      const Domain& r1 = reachable(e);
      const Domain& r2 = reachable(e.reverse());
      if (!is_subset(r1, r2) && !is_subset(r2, r1)) {
        set_marked(e, true);
        set_marked(u, true);
        set_marked(v, true);
        cover.insert(cluster(u).begin(), cluster(u).end());
        cover.insert(cluster(v).begin(), cluster(v).end());
      }
    }

    Domain uncovered;
    for (Arg x : domain) {
      if (!cover.count(x)) {
        uncovered.push_back(x);
      }
    }

    while (!uncovered.empty()) {
      Vertex* v = find_cluster_meets(uncovered);
      assert(v);
      uncovered -= cluster(v);
      set_marked(v, true);
    }
  }

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds a cluster vertex with the supplied domain.
  Vertex* add_vertex(Domain cluster) {
    assert(cluster.is_sorted());
    Vertex* v = UndirectedGraph::add_vertex();
    vertex_annotation(v).value.owner = v;
    vertex_annotation(v).value.reset(std::move(cluster));
    cluster_index_.insert(&vertex_annotation(v).value, vertex_annotation(v).value.hooks);
    return v;
  }

  /// Adds a cluster vertex with a typed property.
  template <typename T = VP>
  Vertex* add_vertex(Domain cluster, T property) requires (!std::is_void_v<T>) {
    Vertex* v = add_vertex(std::move(cluster));
    (*this)[v] = std::move(property);
    return v;
  }

  /// Adds an edge with the supplied separator.
  edge_descriptor add_edge(Vertex* u, Vertex* v, Domain separator) {
    assert(u != v);
    assert(separator.is_sorted());
    assert(is_subset(separator, cluster(u)));
    assert(is_subset(separator, cluster(v)));
    edge_descriptor e = UndirectedGraph::add_edge(u, v);
    edge_annotation(e).value.owner = e.get();
    edge_annotation(e).value.reset(std::move(separator));
    separator_index_.insert(&edge_annotation(e).value, edge_annotation(e).value.hooks);
    reachable_.emplace(e.get(), std::array<Domain, 2>{});
    return e;
  }

  /// Adds an edge using the intersection of endpoint clusters as the separator.
  edge_descriptor add_edge(Vertex* u, Vertex* v) {
    return add_edge(u, v, cluster(u) & cluster(v));
  }

  /// Adds an edge with the supplied separator and typed property.
  template <typename T = EP>
  edge_descriptor add_edge(Vertex* u, Vertex* v, Domain separator, T property) requires (!std::is_void_v<T>) {
    edge_descriptor e = add_edge(u, v, std::move(separator));
    (*this)[e] = std::move(property);
    return e;
  }

  /// Adds an edge with the inferred separator and typed property.
  template <typename T = EP>
  edge_descriptor add_edge(Vertex* u, Vertex* v, T property) requires (!std::is_void_v<T>) {
    edge_descriptor e = add_edge(u, v);
    (*this)[e] = std::move(property);
    return e;
  }

  /// Updates the cluster domain associated with a vertex.
  void update_cluster(Vertex* u, const Domain& cluster) {
    if (this->cluster(u) != cluster) {
      cluster_index_.erase(&vertex_annotation(u).value, vertex_annotation(u).value.hooks);
      vertex_annotation(u).value.owner = u;
      vertex_annotation(u).value.reset(cluster);
      cluster_index_.insert(&vertex_annotation(u).value, vertex_annotation(u).value.hooks);
    }
  }

  /// Updates the separator domain associated with an edge.
  void update_separator(edge_descriptor e, const Domain& separator) {
    if (this->separator(e) != separator) {
      separator_index_.erase(&edge_annotation(e).value, edge_annotation(e).value.hooks);
      edge_annotation(e).value.owner = e.get();
      edge_annotation(e).value.reset(separator);
      separator_index_.insert(&edge_annotation(e).value, edge_annotation(e).value.hooks);
    }
  }

  /// Merges the endpoints of an edge and returns the retained vertex.
  Vertex* merge(edge_descriptor e) {
    Vertex* u = e.source();
    Vertex* v = e.target();

    for (edge_descriptor out : out_edges(u)) {
      if (out.target() != v) {
        edge_descriptor new_edge = add_edge(v, out.target(), separator(out));
        if constexpr (!std::is_void_v<EP>) {
          (*this)[new_edge] = (*this)[out];
        }
      }
    }

    update_cluster(v, cluster(u) | cluster(v));
    remove_vertex(u);
    return v;
  }

  /// Removes a vertex and its associated cluster.
  void remove_vertex(Vertex* u) {
    cluster_index_.erase(&vertex_annotation(u).value, vertex_annotation(u).value.hooks);
    clear_vertex(u);
    UndirectedGraph::remove_vertex(u);
  }

  /// Removes an edge and its associated separator. Returns 1 if removed.
  size_t remove_edge(edge_descriptor e) {
    if (!contains(e)) {
      return 0;
    }
    separator_index_.erase(&edge_annotation(e).value, edge_annotation(e).value.hooks);
    reachable_.erase(e.get());
    return UndirectedGraph::remove_edge(e);
  }

  /// Removes the edge between `u` and `v`. Returns 1 if removed.
  size_t remove_edge(Vertex* u, Vertex* v) {
    if (!contains(u) || !contains(v)) {
      return 0;
    }
    for (edge_descriptor e : out_edges(u)) {
      if (e.target() == v) {
        return remove_edge(e);
      }
    }
    return 0;
  }

  /// Removes all separators incident to `u`.
  void clear_vertex(Vertex* u) {
    std::vector<edge_descriptor> incident(out_edges(u).begin(), out_edges(u).end());
    for (edge_descriptor e : incident) {
      remove_edge(e);
    }
  }

  /// Removes all separators from the graph.
  void remove_edges() {
    std::vector<edge_descriptor> all_edges(edges().begin(), edges().end());
    for (edge_descriptor e : all_edges) {
      remove_edge(e);
    }
  }

  /// Removes all clusters, separators, and index state.
  void clear() {
    UndirectedGraph::clear();
    cluster_index_.clear();
    separator_index_.clear();
    reachable_.clear();
  }

  // Triangulation
  //--------------------------------------------------------------------------
  /// Initializes this graph to the triangulation of a Markov network.
  void triangulated(MarkovNetwork& mn, const EliminationStrategy& strategy) {
    clear();
    mn.eliminate(strategy, [&](MarkovNetwork::vertex_descriptor v) {
      Domain clique(mn.adjacent_vertices(v));
      clique.push_back(v);
      clique.sort();
      if (is_maximal(cluster_index_, clique)) {
        add_vertex(std::move(clique));
      }
    });
    mst_edges();
  }

  /// Initializes this graph from triangulated cliques and adds MST separators.
  std::vector<Vertex*> triangulated(const std::vector<Domain>& cliques) {
    clear();
    std::vector<Vertex*> result;
    result.reserve(cliques.size());
    for (const Domain& clique : cliques) {
      result.push_back(add_vertex(clique));
    }
    mst_edges();
    return result;
  }

  /// Replaces the current edges with a maximum spanning tree over cluster overlaps.
  void mst_edges() {
    remove_edges();
    if (empty()) {
      return;
    }

    Vertex* root = *vertices().begin();
    for (Vertex* u : vertices()) {
      intersecting_clusters(cluster(u), [this, u](Vertex* v) {
        if (u < v) {
          add_edge(u, v);
        }
      });
      if (root != u) {
        add_edge(root, u);
      }
    }

    std::vector<std::pair<Vertex*, Vertex*>> tree_edges;
    auto weight = boost::make_function_property_map<edge_descriptor>([this](edge_descriptor e) {
      return -ptrdiff_t(separator(e).size());
    });

    boost::kruskal_minimum_spanning_tree(
      *this, std::back_inserter(tree_edges),
      boost::weight_map(weight).vertex_index_map(vertex_index_map()));

    remove_edges();
    for (auto [s, t] : tree_edges) {
      add_edge(s, t);
    }
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const UndirectedGraph>(this));

    ar(cereal::make_size_tag(num_vertices()));
    for (Vertex* v : vertices()) {
      ar(cereal::make_nvp("cluster", vertex_annotation(v).value));
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }

    ar(cereal::make_size_tag(num_edges()));
    for (edge_descriptor e : edges()) {
      ar(cereal::make_nvp("separator", edge_annotation(e).value));
      if constexpr (!std::is_void_v<EP>) {
        ar(cereal::make_nvp("property", operator[](e)));
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<UndirectedGraph>(this));

    cereal::size_type vertex_count;
    ar(cereal::make_size_tag(vertex_count));
    assert(vertex_count == num_vertices());

    cluster_index_.clear();
    for (Vertex* v : vertices()) {
      ar(cereal::make_nvp("cluster", vertex_annotation(v).value));
      vertex_annotation(v).value.owner = v;
      cluster_index_.insert(&vertex_annotation(v).value, vertex_annotation(v).value.hooks);
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    assert(edge_count == num_edges());

    separator_index_.clear();
    reachable_.clear();
    for (edge_descriptor e : edges()) {
      ar(cereal::make_nvp("separator", edge_annotation(e).value));
      edge_annotation(e).value.owner = e.get();
      separator_index_.insert(&edge_annotation(e).value, edge_annotation(e).value.hooks);
      reachable_.emplace(e.get(), std::array<Domain, 2>{});
      if constexpr (!std::is_void_v<EP>) {
        ar(cereal::make_nvp("property", operator[](e)));
      }
    }
  }

private:
  class ReachableVisitor {
  public:
    ReachableVisitor(ClusterGraph& graph, bool propagate_past_empty, const ankerl::unordered_dense::set<Arg>* filter = nullptr)
      : graph_(graph),
        propagate_past_empty_(propagate_past_empty),
        filter_(filter) {}

    void operator()(edge_descriptor e) const {
      Domain r;
      if (!graph_.separator(e).empty() || propagate_past_empty_) {
        for (Arg x : graph_.cluster(e.source())) {
          if (!filter_ || filter_->count(x)) {
            r.push_back(x);
          }
        }

        for (edge_descriptor out : graph_.out_edges(e.source())) {
          if (out.target() != e.target()) {
            r.append(graph_.reachable_.at(out.get())[!out.index()]);
          }
        }
        r.unique();
      }

      graph_.reachable_[e.get()][e.index()] = std::move(r);
    }

  private:
    ClusterGraph& graph_;
    bool propagate_past_empty_;
    const ankerl::unordered_dense::set<Arg>* filter_;
  };

  VertexAnnotation& vertex_annotation(Vertex* v) {
    return opaque_cast<VertexAnnotation>(UndirectedGraph::property(v));
  }

  const VertexAnnotation& vertex_annotation(Vertex* v) const {
    return opaque_cast<VertexAnnotation>(UndirectedGraph::property(v));
  }

  EdgeAnnotation& edge_annotation(edge_descriptor e) {
    return opaque_cast<EdgeAnnotation>(UndirectedGraph::property(e));
  }

  const EdgeAnnotation& edge_annotation(edge_descriptor e) const {
    return opaque_cast<EdgeAnnotation>(UndirectedGraph::property(e));
  }

  void rebuild_indices() {
    cluster_index_.clear();
    separator_index_.clear();
    reachable_.clear();

    for (Vertex* v : vertices()) {
      vertex_annotation(v).value.owner = v;
      cluster_index_.insert(&vertex_annotation(v).value, vertex_annotation(v).value.hooks);
    }
    for (edge_descriptor e : edges()) {
      edge_annotation(e).value.owner = e.get();
      separator_index_.insert(&edge_annotation(e).value, edge_annotation(e).value.hooks);
      reachable_.emplace(e.get(), std::array<Domain, 2>{});
    }
  }

  void copy_reachable(const ClusterGraph& other) {
    auto src = other.edges().begin();
    auto dst = edges().begin();
    for (; src != other.edges().end() && dst != edges().end(); ++src, ++dst) {
      reachable_[*dst] = other.reachable_.at(*src);
    }
  }

  DomainIndex<VertexDomain> cluster_index_;
  DomainIndex<EdgeDomain> separator_index_;
  ReachableMap reachable_;
};

} // namespace libgm
