#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/model/markov_structure.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>

#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

/**
 * A Markov network whose graph structure is implemented by `UndirectedGraph`
 * and whose model-specific layer maps arguments to graph vertices.
 *
 * Vertices store `Annotated<Arg, VP>` and edges store `EP`. Both `VP` and
 * `EP` may be `void`.
 */
template <Argument Arg, typename VP = void, typename EP = VP>
class MarkovNetwork : private UndirectedGraph {
  using VertexAnnotation = Annotated<Arg, VP>;

public:
  using argument_type = Arg;
  using domain_type = Domain<Arg>;
  using structure_type = MarkovStructure<Arg>;
  using Vertex = UndirectedGraph::Vertex;
  using Edge = UndirectedGraph::Edge;
  using VertexMap = ankerl::unordered_dense::map<Arg, Vertex*>;

  using UndirectedGraph::adjacency_iterator;
  using UndirectedGraph::adjacent_vertices;
  using UndirectedGraph::contains;
  using UndirectedGraph::degree_size_type;
  using UndirectedGraph::directed_category;
  using UndirectedGraph::edge_descriptor;
  using UndirectedGraph::edge_iterator;
  using UndirectedGraph::edge_parallel_category;
  using UndirectedGraph::edges;
  using UndirectedGraph::edges_size_type;
  using UndirectedGraph::empty;
  using UndirectedGraph::in_degree;
  using UndirectedGraph::in_edge_iterator;
  using UndirectedGraph::in_edges;
  using UndirectedGraph::num_edges;
  using UndirectedGraph::num_vertices;
  using UndirectedGraph::out_degree;
  using UndirectedGraph::out_edge_iterator;
  using UndirectedGraph::out_edges;
  using UndirectedGraph::property;
  using UndirectedGraph::traversal_category;
  using UndirectedGraph::vertex_descriptor;
  using UndirectedGraph::vertex_index_map;
  using UndirectedGraph::vertex_iterator;
  using UndirectedGraph::vertices;
  using UndirectedGraph::vertices_size_type;

  using argument_iterator = MapKeyIterator<VertexMap>;
  using vertex_property_reference = std::add_lvalue_reference_t<VP>;
  using const_vertex_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;
  using edge_property_reference = std::add_lvalue_reference_t<EP>;
  using const_edge_property_reference = std::add_lvalue_reference_t<std::add_const_t<EP>>;

  explicit MarkovNetwork(size_t count = 0)
    : UndirectedGraph(property_layout<VertexAnnotation>(), property_layout<EP>()) {
    vertices_.reserve(count);
  }

  MarkovNetwork(const MarkovNetwork& other)
    : UndirectedGraph(other) {
    rebuild_map();
  }

  MarkovNetwork(MarkovNetwork&& other) noexcept = default;

  MarkovNetwork& operator=(const MarkovNetwork& other) {
    if (this != &other) {
      UndirectedGraph::operator=(other);
      rebuild_map();
    }
    return *this;
  }

  MarkovNetwork& operator=(MarkovNetwork&& other) noexcept = default;

  static Vertex* null_vertex() {
    return UndirectedGraph::null_vertex();
  }

  std::ranges::subrange<argument_iterator> arguments() const {
    return {vertices_.begin(), vertices_.end()};
  }

  std::ranges::subrange<out_edge_iterator> out_edges(Arg u) const {
    return UndirectedGraph::out_edges(vertex(u));
  }

  std::ranges::subrange<in_edge_iterator> in_edges(Arg u) const {
    return UndirectedGraph::in_edges(vertex(u));
  }

  std::ranges::subrange<adjacency_iterator> adjacent_vertices(Arg u) const {
    return UndirectedGraph::adjacent_vertices(vertex(u));
  }

  Vertex* vertex(Arg u) const {
    return vertices_.at(u);
  }

  Arg argument(Vertex* v) const {
    return annotation(v).value;
  }

  domain_type domain(edge_descriptor e) const {
    return {argument(e.source()), argument(e.target())};
  }

  bool is_nominal(edge_descriptor e) const {
    return !e.index();
  }

  bool contains(Arg u) const {
    return vertices_.contains(u);
  }

  bool contains(Arg u, Arg v) const {
    return contains(u) && contains(v) && UndirectedGraph::contains(vertex(u), vertex(v));
  }

  edge_descriptor edge(Arg u, Arg v) const {
    if (!contains(u, v)) {
      return {};
    }
    for (edge_descriptor e : out_edges(u)) {
      if (e.target() == vertex(v)) {
        return e;
      }
    }
    return {};
  }

  size_t degree(Arg u) const {
    return UndirectedGraph::degree(vertex(u));
  }

  size_t in_degree(Arg u) const {
    return UndirectedGraph::in_degree(vertex(u));
  }

  size_t out_degree(Arg u) const {
    return UndirectedGraph::out_degree(vertex(u));
  }

  OpaqueRef property(Arg u) {
    return UndirectedGraph::property(vertex(u));
  }

  OpaqueCref property(Arg u) const {
    return UndirectedGraph::property(vertex(u));
  }

  vertex_property_reference operator[](Arg u) {
    return annotation(vertex(u)).property();
  }

  const_vertex_property_reference operator[](Arg u) const {
    return annotation(vertex(u)).property();
  }

  vertex_property_reference operator[](Vertex* v) {
    return annotation(v).property();
  }

  const_vertex_property_reference operator[](Vertex* v) const {
    return annotation(v).property();
  }

  edge_property_reference operator[](edge_descriptor e) {
    return opaque_cast<EP>(UndirectedGraph::property(e));
  }

  const_edge_property_reference operator[](edge_descriptor e) const {
    return opaque_cast<EP>(UndirectedGraph::property(e));
  }

  bool add_vertex(Arg u) {
    if (contains(u)) {
      return false;
    }
    Vertex* v = UndirectedGraph::add_vertex();
    annotation(v).value = u;
    vertices_.emplace(u, v);
    return true;
  }

  template <typename T = VP>
  bool add_vertex(Arg u, T property) requires (!std::is_void_v<T>) {
    bool inserted = add_vertex(u);
    if (inserted) {
      (*this)[u] = std::move(property);
    }
    return inserted;
  }

  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v) {
    if (v < u) {
      throw std::invalid_argument("MarkovNetwork::add_edge: arguments must be in canonical order");
    }
    if (!contains(u)) {
      add_vertex(u);
    }
    if (!contains(v)) {
      add_vertex(v);
    }
    edge_descriptor existing = edge(u, v);
    if (existing) {
      return {existing, false};
    }
    return {UndirectedGraph::add_edge(vertex(u), vertex(v)), true};
  }

  template <typename T = EP>
  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v, T property) requires (!std::is_void_v<T>) {
    auto result = add_edge(u, v);
    if (result.second) {
      (*this)[result.first] = std::move(property);
    }
    return result;
  }

  void add_edges(Arg u, const std::vector<Arg>& vs) {
    for (const Arg& v : vs) {
      add_edge(u, v);
    }
  }

  void add_clique(const domain_type& args) {
    for (const Arg& u : args) {
      add_vertex(u);
    }
    for (auto it1 = args.begin(); it1 != args.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != args.end(); ++it2) {
        add_edge(*it1, *it2);
      }
    }
  }

  size_t remove_vertex(Arg u) {
    auto it = vertices_.find(u);
    if (it == vertices_.end()) {
      return 0;
    }
    UndirectedGraph::remove_vertex(it->second);
    vertices_.erase(it);
    return 1;
  }

  size_t remove_edge(Arg u, Arg v) {
    return contains(u, v) ? UndirectedGraph::remove_edge(vertex(u), vertex(v)) : 0;
  }

  void remove_edges(Arg u) {
    UndirectedGraph::clear_vertex(vertex(u));
  }

  using UndirectedGraph::remove_edges;

  void clear() {
    UndirectedGraph::clear();
    vertices_.clear();
  }

  void init_vertices(const std::function<VP(Arg)>& init_fn) requires (!std::is_void_v<VP>) {
    for (auto v : vertices()) {
      (*this)[v] = init_fn(argument(v));
    }
  }

  void init_edges(const std::function<EP(edge_descriptor)>& init_fn) requires (!std::is_void_v<EP>) {
    for (edge_descriptor e : edges()) {
      (*this)[e] = init_fn(e);
    }
  }

  structure_type structure() const {
    auto* self = const_cast<MarkovNetwork*>(this);
    auto index_map = self->vertex_index_map();
    structure_type result;
    for (auto v : vertices()) {
      result.add_vertex(argument(v));
    }
    for (edge_descriptor e : edges()) {
      result.add_clique({get(index_map, e.source()), get(index_map, e.target())});
    }
    return result;
  }

  const UndirectedGraph& graph() const {
    return *this;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const UndirectedGraph>(this));

    ar(cereal::make_size_tag(num_vertices()));
    for (auto v : vertices()) {
      ar(cereal::make_nvp("argument", annotation(v).value));
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("vertex_property", operator[](v)));
      }
    }

    ar(cereal::make_size_tag(num_edges()));
    for (edge_descriptor e : edges()) {
      if constexpr (!std::is_void_v<EP>) {
        ar(cereal::make_nvp("edge_property", operator[](e)));
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<UndirectedGraph>(this));

    cereal::size_type vertex_count;
    ar(cereal::make_size_tag(vertex_count));
    assert(vertex_count == num_vertices());

    vertices_.clear();
    vertices_.reserve(vertex_count);
    for (auto v : vertices()) {
      Arg u;
      ar(cereal::make_nvp("argument", u));
      annotation(v).value = u;
      vertices_.emplace(u, v);
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("vertex_property", operator[](v)));
      }
    }

    cereal::size_type edge_count;
    ar(cereal::make_size_tag(edge_count));
    assert(edge_count == num_edges());
    for (edge_descriptor e : edges()) {
      if constexpr (!std::is_void_v<EP>) {
        ar(cereal::make_nvp("edge_property", operator[](e)));
      }
    }
  }

private:
  VertexAnnotation& annotation(Vertex* v) {
    return opaque_cast<VertexAnnotation>(UndirectedGraph::property(v));
  }

  const VertexAnnotation& annotation(Vertex* v) const {
    return opaque_cast<VertexAnnotation>(UndirectedGraph::property(v));
  }

  void rebuild_map() {
    vertices_.clear();
    vertices_.reserve(num_vertices());
    for (auto v : vertices()) {
      vertices_.emplace(argument(v), v);
    }
  }

  VertexMap vertices_;
};

} // namespace libgm
