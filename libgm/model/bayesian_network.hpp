#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/graph/directed_graph.hpp>
#include <libgm/model/markov_structure.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

template <typename VP = void>
class BayesianNetwork : private DirectedGraph {
  using Annotation = Annotated<Arg, VP>;
  using VertexMap = ankerl::unordered_dense::map<Arg, DirectedGraph::Vertex*>;

public:
  using DirectedGraph::Vertex;
  using DirectedGraph::adjacency_iterator;
  using DirectedGraph::adjacent_vertices;
  using DirectedGraph::contains;
  using DirectedGraph::compute_indices;
  using DirectedGraph::degree;
  using DirectedGraph::edge;
  using DirectedGraph::edge_descriptor;
  using DirectedGraph::empty;
  using DirectedGraph::index;
  using DirectedGraph::in_degree;
  using DirectedGraph::in_edges;
  using DirectedGraph::in_edge_iterator;
  using DirectedGraph::indices;
  using DirectedGraph::null_vertex;
  using DirectedGraph::num_edges;
  using DirectedGraph::num_vertices;
  using DirectedGraph::out_degree;
  using DirectedGraph::out_edges;
  using DirectedGraph::out_edge_iterator;
  using DirectedGraph::parents;
  using DirectedGraph::vertex_descriptor;
  using DirectedGraph::vertex_iterator;
  using DirectedGraph::vertices;

  using property_reference = std::add_lvalue_reference_t<VP>;
  using const_property_reference = std::add_lvalue_reference_t<std::add_const_t<VP>>;

  explicit BayesianNetwork(size_t count = 0)
    : DirectedGraph(property_layout<Annotation>()),
      vertices_(count) {}

  BayesianNetwork(const BayesianNetwork& other)
    : DirectedGraph(other) {
    rebuild_map();
  }

  BayesianNetwork(BayesianNetwork&& other) noexcept = default;

  BayesianNetwork& operator=(const BayesianNetwork& other) {
    if (this != &other) {
      DirectedGraph::operator=(other);
      rebuild_map();
    }
    return *this;
  }

  BayesianNetwork& operator=(BayesianNetwork&& other) noexcept = default;

  using DirectedGraph::property;

  bool contains(Arg u) const {
    return vertices_.contains(u);
  }

  Vertex* vertex(Arg u) const {
    return vertices_.at(u);
  }

  Arg argument(Vertex* u) const {
    return annotated(u).value;
  }

  Domain arguments(Vertex* u) const {
    Domain result;
    result.reserve(DirectedGraph::parents(u).size() + 1);
    result.push_back(argument(u));
    for (Vertex* parent : DirectedGraph::parents(u)) {
      result.push_back(argument(parent));
    }
    return result;
  }

  property_reference operator[](Vertex* u) {
    return annotated(u).property();
  }

  const_property_reference operator[](Vertex* u) const {
    return annotated(u).property();
  }

  property_reference operator[](Arg u) {
    return operator[](vertex(u));
  }

  const_property_reference operator[](Arg u) const {
    return operator[](vertex(u));
  }

  Vertex* add_vertex(Arg u, const Domain& parents) {
    if (contains(u)) {
      throw std::invalid_argument("BayesianNetwork::add_vertex: vertex already exists");
    }

    for (Arg parent : parents) {
      if (parent == u) {
        throw std::invalid_argument("BayesianNetwork::add_vertex: self-parent is not allowed");
      }
      if (!contains(parent)) {
        throw std::out_of_range("BayesianNetwork::add_vertex: parent not found");
      }
    }

    std::vector<Vertex*> parent_vertices;
    parent_vertices.reserve(parents.size());
    for (Arg parent : parents) {
      parent_vertices.push_back(vertex(parent));
    }

    Vertex* v = DirectedGraph::add_vertex(std::move(parent_vertices));
    annotated(v).value = u;
    vertices_.emplace(u, v);
    return v;
  }

  template <typename T = VP>
  Vertex* add_vertex(Arg u, const Domain& parents, T vp) requires (!std::is_void_v<T>) {
    Vertex* v = add_vertex(u, parents);
    (*this)[v] = std::move(vp);
    return v;
  }

  void set_parents(Arg u, const Domain& parents) {
    Vertex* v = vertex(u);

    for (Arg parent : parents) {
      if (parent == u) {
        throw std::invalid_argument("BayesianNetwork::set_parents: self-parent is not allowed");
      }
      if (!contains(parent)) {
        throw std::out_of_range("BayesianNetwork::set_parents: parent not found");
      }
    }

    std::vector<Vertex*> parent_vertices;
    parent_vertices.reserve(parents.size());
    for (Arg parent : parents) {
      parent_vertices.push_back(vertex(parent));
    }

    DirectedGraph::set_parents(v, std::move(parent_vertices));
  }

  size_t remove_vertex(Arg u) {
    auto it = vertices_.find(u);
    if (it == vertices_.end()) {
      return 0;
    }
    size_t result = DirectedGraph::remove_vertex(it->second);
    vertices_.erase(it);
    return result;
  }

  void remove_in_edges(Arg u) {
    DirectedGraph::remove_in_edges(vertex(u));
  }

  void clear() {
    DirectedGraph::clear();
    vertices_.clear();
  }

  MarkovStructure markov_structure() const {
    MarkovStructure mg;
    compute_indices();
    for (Vertex* v : vertices()) {
      mg.add_vertex(argument(v));
    }
    for (Vertex* v : vertices()) {
      mg.add_clique(indices(v));
    }
    return mg;
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const DirectedGraph>(this));
    ar(cereal::make_size_tag(num_vertices()));
    for (Vertex* v : vertices()) {
      ar(cereal::make_nvp("argument", annotated(v).value));
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<DirectedGraph>(this));

    cereal::size_type n;
    ar(cereal::make_size_tag(n));
    assert(n == num_vertices());

    vertices_.clear();
    vertices_.reserve(n);
    for (Vertex* v : vertices()) {
      Arg u;
      ar(cereal::make_nvp("argument", u));
      annotated(v).value = u;
      vertices_.emplace(u, v);
      if constexpr (!std::is_void_v<VP>) {
        ar(cereal::make_nvp("property", operator[](v)));
      }
    }
  }

private:
  Annotation& annotated(Vertex* u) {
    return opaque_cast<Annotation>(DirectedGraph::property(u));
  }

  const Annotation& annotated(Vertex* u) const {
    return opaque_cast<Annotation>(DirectedGraph::property(u));
  }

  void rebuild_map() {
    vertices_.clear();
    vertices_.reserve(num_vertices());
    for (Vertex* v : vertices()) {
      vertices_.emplace(argument(v), v);
    }
  }

  VertexMap vertices_;
};

} // namespace libgm
