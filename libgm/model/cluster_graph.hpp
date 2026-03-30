#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/domain_index.hpp>
#include <libgm/datastructure/domain_index_operations.hpp>
#include <libgm/datastructure/indexed_domain.hpp>
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
#include <cassert>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

template <typename VP = void, typename EP = VP>
class ClusterGraph : private UndirectedGraph {
  using VertexDomain = IndexedDomain<UndirectedGraph::Vertex>;
  using EdgeDomain = IndexedDomain<UndirectedGraph::Edge>;
  using VertexAnnotation = Annotated<VertexDomain, VP>;
  using EdgeAnnotation = Annotated<EdgeDomain, EP>;

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
  using UndirectedGraph::remove_edge;
  using UndirectedGraph::remove_edges;
  using UndirectedGraph::remove_vertex;
  using UndirectedGraph::root;
  using UndirectedGraph::traversal_category;
  using UndirectedGraph::vertex_color_map;
  using UndirectedGraph::vertex_descriptor;
  using UndirectedGraph::vertex_index_map;
  using UndirectedGraph::vertex_iterator;
  using UndirectedGraph::vertices_size_type;
  using UndirectedGraph::vertices;
  using UndirectedGraph::clear_vertex;
  /// Iterator over arguments present in the cluster index.
  using argument_iterator = DomainIndex<Vertex>::argument_iterator;

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
  }

  ClusterGraph(ClusterGraph&& other) noexcept = default;

  ClusterGraph& operator=(const ClusterGraph& other) {
    if (this != &other) {
      UndirectedGraph::operator=(other);
      rebuild_indices();
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
      Vertex* v = cluster_index_[x];

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
    return find_min_cover(cluster_index_, dom);
  }

  /// Returns a separator whose domain covers `dom`, or the null edge.
  edge_descriptor find_separator_cover(const Domain& dom) const {
    Edge* edge = find_min_cover(separator_index_, dom);
    return edge ? edge_descriptor(edge) : edge_descriptor();
  }

  /// Returns a cluster whose domain best intersects `dom`.
  Vertex* find_cluster_meets(const Domain& dom) const {
    return find_max_intersection(cluster_index_, dom);
  }

  /// Returns a separator whose domain best intersects `dom`.
  edge_descriptor find_separator_meets(const Domain& dom) const {
    Edge* edge = find_max_intersection(separator_index_, dom);
    return edge ? edge_descriptor(edge) : edge_descriptor();
  }

  /// Visits all clusters intersecting `dom`.
  void intersecting_clusters(const Domain& dom, VertexVisitor visitor) const {
    visit_intersections(cluster_index_, dom, std::move(visitor));
  }

  /// Visits all separators intersecting `dom`.
  void intersecting_separators(const Domain& dom, ModelEdgeVisitor visitor) const {
    visit_intersections(separator_index_, dom, [&](Edge* edge) {
      visitor(edge_descriptor(edge));
    });
  }

  // Modifications
  //--------------------------------------------------------------------------
  /// Adds a cluster vertex with the supplied domain.
  Vertex* add_vertex(Domain cluster) {
    assert(cluster.is_sorted());
    Vertex* v = UndirectedGraph::add_vertex();
    vertex_annotation(v).value.owner = v;
    vertex_annotation(v).value.reset(std::move(cluster));
    cluster_index_.insert(vertex_annotation(v).value);
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
    separator_index_.insert(edge_annotation(e).value);
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
      vertex_annotation(u).value.owner = u;
      vertex_annotation(u).value.reset(cluster);
      cluster_index_.insert(vertex_annotation(u).value);
    }
  }

  /// Updates the separator domain associated with an edge.
  void update_separator(edge_descriptor e, const Domain& separator) {
    if (this->separator(e) != separator) {
      edge_annotation(e).value.owner = e.get();
      edge_annotation(e).value.reset(separator);
      separator_index_.insert(edge_annotation(e).value);
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

  /// Removes all clusters, separators, and index state.
  void clear() {
    UndirectedGraph::clear();
    cluster_index_.clear();
    separator_index_.clear();
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
      ar(vertex_annotation(v).value);
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }

    ar(cereal::make_size_tag(num_edges()));
    for (edge_descriptor e : edges()) {
      ar(edge_annotation(e).value);
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
      VertexDomain& indexed_cluster = vertex_annotation(v).value;
      ar(indexed_cluster);
      indexed_cluster.owner = v;
      cluster_index_.insert(indexed_cluster);
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    assert(edge_count == num_edges());

    separator_index_.clear();
    for (edge_descriptor e : edges()) {
      EdgeDomain& indexed_separator = edge_annotation(e).value;
      ar(indexed_separator);
      indexed_separator.owner = e.get();
      separator_index_.insert(indexed_separator);
      if constexpr (!std::is_void_v<EP>) {
        ar(cereal::make_nvp("property", operator[](e)));
      }
    }
  }

private:
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

    for (Vertex* v : vertices()) {
      vertex_annotation(v).value.owner = v;
      cluster_index_.insert(vertex_annotation(v).value);
    }
    for (edge_descriptor e : edges()) {
      edge_annotation(e).value.owner = e.get();
      separator_index_.insert(edge_annotation(e).value);
    }
  }

  DomainIndex<Vertex> cluster_index_;
  DomainIndex<Edge> separator_index_;
};

} // namespace libgm
