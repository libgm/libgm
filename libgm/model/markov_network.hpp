#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/model/markov_structure.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_key_iterator.hpp>

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
template <typename VP = void, typename EP = VP>
class MarkovNetwork : private UndirectedGraph {
  using VertexAnnotation = Annotated<Arg, VP>;
  using VertexMap = ankerl::unordered_dense::map<Arg, Vertex*>;

public:
  // Base graph types
  //--------------------------------------------------------------------------
  using UndirectedGraph::Vertex;
  using UndirectedGraph::Edge;
  using UndirectedGraph::vertex_descriptor;
  using UndirectedGraph::edge_descriptor;

  // Base graph categories and iterators
  //--------------------------------------------------------------------------
  using UndirectedGraph::adjacent_vertices;
  using UndirectedGraph::contains;
  using UndirectedGraph::adjacency_iterator;
  using UndirectedGraph::degree_size_type;
  using UndirectedGraph::directed_category;
  using UndirectedGraph::edge_iterator;
  using UndirectedGraph::edge_parallel_category;
  using UndirectedGraph::edges;
  using UndirectedGraph::edges_size_type;
  using UndirectedGraph::empty;
  using UndirectedGraph::in_degree;
  using UndirectedGraph::in_edges;
  using UndirectedGraph::in_edge_iterator;
  using UndirectedGraph::num_edges;
  using UndirectedGraph::num_vertices;
  using UndirectedGraph::out_degree;
  using UndirectedGraph::out_edges;
  using UndirectedGraph::out_edge_iterator;
  using UndirectedGraph::property;
  using UndirectedGraph::traversal_category;
  using UndirectedGraph::vertex_index_map;
  using UndirectedGraph::vertex_iterator;
  using UndirectedGraph::vertices;
  using UndirectedGraph::vertices_size_type;

  // Model-specific types
  //--------------------------------------------------------------------------
  /// Iterator over all arguments in the network.
  using argument_iterator = MapKeyIterator<VertexMap>;
  /// Mutable reference to a vertex property.
  using vertex_property_reference = std::add_lvalue_reference_t<VP>;
  /// Const reference to a vertex property.
  using const_vertex_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;
  /// Mutable reference to an edge property.
  using edge_property_reference = std::add_lvalue_reference_t<EP>;
  /// Const reference to an edge property.
  using const_edge_property_reference = std::add_lvalue_reference_t<std::add_const_t<EP>>;

  /// Constructs an empty network and reserves space for `count` arguments.
  explicit MarkovNetwork(size_t count = 0)
    : UndirectedGraph(property_layout<VertexAnnotation>(), property_layout<EP>()) {
    vertices_.reserve(count);
  }

  /// Copy-constructs the network and rebuilds the argument map.
  MarkovNetwork(const MarkovNetwork& other)
    : UndirectedGraph(other) {
    rebuild_map();
  }

  /// Move-constructs the network.
  MarkovNetwork(MarkovNetwork&& other) noexcept = default;

  /// Copy-assigns the network and rebuilds the argument map.
  MarkovNetwork& operator=(const MarkovNetwork& other) {
    if (this != &other) {
      UndirectedGraph::operator=(other);
      rebuild_map();
    }
    return *this;
  }

  /// Move-assigns the network.
  MarkovNetwork& operator=(MarkovNetwork&& other) noexcept = default;

  /// Returns the null vertex descriptor.
  static Vertex* null_vertex() {
    return UndirectedGraph::null_vertex();
  }

  /// Returns the range of all arguments present in the network.
  std::ranges::subrange<argument_iterator> arguments() const {
    return {vertices_.begin(), vertices_.end()};
  }

  /// Returns the outgoing directed edge views adjacent to the given argument.
  std::ranges::subrange<out_edge_iterator> out_edges(Arg u) const {
    return UndirectedGraph::out_edges(vertex(u));
  }

  /// Returns the incoming directed edge views adjacent to the given argument.
  std::ranges::subrange<in_edge_iterator> in_edges(Arg u) const {
    return UndirectedGraph::in_edges(vertex(u));
  }

  /// Returns the arguments adjacent to the given argument.
  std::ranges::subrange<adjacency_iterator> adjacent_vertices(Arg u) const {
    return UndirectedGraph::adjacent_vertices(vertex(u));
  }

  /// Returns the vertex associated with the given argument.
  Vertex* vertex(Arg u) const {
    return vertices_.at(u);
  }

  /// Returns the argument associated with the given vertex.
  Arg argument(Vertex* v) const {
    return annotation(v).value;
  }

  /// Returns the ordered pair of arguments corresponding to the directed edge view.
  Domain domain(edge_descriptor e) const {
    return {argument(e.source()), argument(e.target())};
  }

  /// Returns true if the directed edge view follows the stored edge orientation.
  bool is_nominal(edge_descriptor e) const {
    return !e.index();
  }

  /// Returns true if the network contains the given argument.
  bool contains(Arg u) const {
    return vertices_.contains(u);
  }

  /// Returns true if the network contains an edge between the given arguments.
  bool contains(Arg u, Arg v) const {
    return contains(u) && contains(v) && UndirectedGraph::contains(vertex(u), vertex(v));
  }

  /// Returns the directed edge view from `u` to `v`, or the null edge if absent.
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

  /// Returns the degree of the given argument.
  size_t degree(Arg u) const {
    return UndirectedGraph::degree(vertex(u));
  }

  /// Returns the in-degree of the given argument.
  size_t in_degree(Arg u) const {
    return UndirectedGraph::in_degree(vertex(u));
  }

  /// Returns the out-degree of the given argument.
  size_t out_degree(Arg u) const {
    return UndirectedGraph::out_degree(vertex(u));
  }

  /// Returns the opaque property associated with the given argument.
  OpaqueRef property(Arg u) {
    return UndirectedGraph::property(vertex(u));
  }

  /// Returns the opaque property associated with the given argument.
  OpaqueCref property(Arg u) const {
    return UndirectedGraph::property(vertex(u));
  }

  /// Returns the typed property associated with the given argument.
  vertex_property_reference operator[](Arg u) {
    return annotation(vertex(u)).property();
  }

  const_vertex_property_reference operator[](Arg u) const {
    return annotation(vertex(u)).property();
  }

  /// Returns the typed property associated with the given vertex.
  vertex_property_reference operator[](Vertex* v) {
    return annotation(v).property();
  }

  const_vertex_property_reference operator[](Vertex* v) const {
    return annotation(v).property();
  }

  /// Returns the typed property associated with the given edge.
  edge_property_reference operator[](edge_descriptor e) {
    return opaque_cast<EP>(UndirectedGraph::property(e));
  }

  const_edge_property_reference operator[](edge_descriptor e) const {
    return opaque_cast<EP>(UndirectedGraph::property(e));
  }

  /// Adds an argument if absent and returns whether it was inserted.
  bool add_vertex(Arg u) {
    if (contains(u)) {
      return false;
    }
    Vertex* v = UndirectedGraph::add_vertex();
    annotation(v).value = u;
    vertices_.emplace(u, v);
    return true;
  }

  /// Adds a typed vertex property together with the argument.
  template <typename T = VP>
  bool add_vertex(Arg u, T property) requires (!std::is_void_v<T>) {
    bool inserted = add_vertex(u);
    if (inserted) {
      (*this)[u] = std::move(property);
    }
    return inserted;
  }

  /// Adds an edge between the given arguments in canonical argument order.
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

  /// Adds an edge property together with the edge.
  template <typename T = EP>
  std::pair<edge_descriptor, bool> add_edge(Arg u, Arg v, T property) requires (!std::is_void_v<T>) {
    auto result = add_edge(u, v);
    if (result.second) {
      (*this)[result.first] = std::move(property);
    }
    return result;
  }

  /// Adds edges between `u` and all arguments in `vs`.
  void add_edges(Arg u, const std::vector<Arg>& vs) {
    for (Arg v : vs) {
      add_edge(u, v);
    }
  }

  /// Adds all edges induced by the given clique.
  void add_clique(const Domain& args) {
    for (Arg u : args) {
      add_vertex(u);
    }
    for (auto it1 = args.begin(); it1 != args.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != args.end(); ++it2) {
        add_edge(*it1, *it2);
      }
    }
  }

  /// Removes the given argument and all of its incident edges.
  size_t remove_vertex(Arg u) {
    auto it = vertices_.find(u);
    if (it == vertices_.end()) {
      return 0;
    }
    UndirectedGraph::remove_vertex(it->second);
    vertices_.erase(it);
    return 1;
  }

  /// Removes the edge between `u` and `v` if present.
  size_t remove_edge(Arg u, Arg v) {
    return contains(u, v) ? UndirectedGraph::remove_edge(vertex(u), vertex(v)) : 0;
  }

  /// Removes all edges incident to `u`.
  void remove_edges(Arg u) {
    UndirectedGraph::clear_vertex(vertex(u));
  }

  using UndirectedGraph::remove_edges;

  /// Removes all arguments and edges from the network.
  void clear() {
    UndirectedGraph::clear();
    vertices_.clear();
  }

  /// Initializes all vertex properties from the supplied generator.
  void init_vertices(const std::function<VP(Arg)>& init_fn) requires (!std::is_void_v<VP>) {
    for (auto v : vertices()) {
      (*this)[v] = init_fn(argument(v));
    }
  }

  /// Initializes all edge properties from the supplied generator.
  void init_edges(const std::function<EP(edge_descriptor)>& init_fn) requires (!std::is_void_v<EP>) {
    for (edge_descriptor e : edges()) {
      (*this)[e] = init_fn(e);
    }
  }

  /// Returns the property-free graph view used for elimination.
  MarkovStructure structure() const {
    MarkovNetwork* self = const_cast<MarkovNetwork*>(this);
    auto index_map = self->vertex_index_map();
    MarkovStructure result;
    for (auto v : vertices()) {
      result.add_vertex(argument(v));
    }
    for (edge_descriptor e : edges()) {
      result.add_clique({get(index_map, e.source()), get(index_map, e.target())});
    }
    return result;
  }

  /// Returns the underlying generic graph.
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
