#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/graph/bipartite_graph.hpp>
#include <libgm/graph/util/property_layout.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/model/markov_network.hpp>
#include <libgm/model/markov_structure.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>

#include <cassert>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace libgm {

template <Argument Arg, typename AP = void, typename FP = void>
class FactorGraph : private BipartiteGraph {
  using ArgumentAnnotation = Annotated<Arg, AP>;
  using FactorAnnotation = Annotated<Domain<Arg>, FP>;
  using ArgumentMap = ankerl::unordered_dense::map<Arg, BipartiteGraph::Vertex1*>;

public:
  using argument_type = Arg;
  using domain_type = Domain<Arg>;
  using markov_structure_type = MarkovStructure<Arg>;
  using Vertex1 = BipartiteGraph::Vertex1;
  using Vertex2 = BipartiteGraph::Vertex2;
  using Argument = Vertex1;
  using Factor = Vertex2;

  using BipartiteGraph::contains;
  using BipartiteGraph::degree;
  using BipartiteGraph::edge12_descriptor;
  using BipartiteGraph::edge21_descriptor;
  using BipartiteGraph::empty;
  using BipartiteGraph::in_edge1_iterator;
  using BipartiteGraph::in_edge2_iterator;
  using BipartiteGraph::in_edges;
  using BipartiteGraph::index;
  using BipartiteGraph::neighbors;
  using BipartiteGraph::num_vertices1;
  using BipartiteGraph::num_vertices2;
  using BipartiteGraph::out_edge1_iterator;
  using BipartiteGraph::out_edge2_iterator;
  using BipartiteGraph::out_edges;
  using BipartiteGraph::property;
  using BipartiteGraph::remove_vertex2;
  using BipartiteGraph::vertex1_descriptor;
  using BipartiteGraph::vertex1_iterator;
  using BipartiteGraph::vertex2_descriptor;
  using BipartiteGraph::vertex2_iterator;
  using BipartiteGraph::vertices1;
  using BipartiteGraph::vertices2;

  using argument_iterator = MapKeyIterator<ArgumentMap>;
  using factor_iterator = vertex2_iterator;
  using argument_property_reference = std::add_lvalue_reference_t<AP>;
  using const_argument_property_reference = std::add_lvalue_reference_t<std::add_const_t<AP>>;
  using factor_property_reference = std::add_lvalue_reference_t<FP>;
  using const_factor_property_reference = std::add_lvalue_reference_t<std::add_const_t<FP>>;

  FactorGraph()
    : BipartiteGraph(property_layout<ArgumentAnnotation>(), property_layout<FactorAnnotation>()) {}

  FactorGraph(const FactorGraph& other)
    : BipartiteGraph(other) {
    rebuild_map();
  }

  FactorGraph(FactorGraph&& other) noexcept = default;

  FactorGraph& operator=(const FactorGraph& other) {
    if (this != &other) {
      BipartiteGraph::operator=(other);
      rebuild_map();
    }
    return *this;
  }

  FactorGraph& operator=(FactorGraph&& other) noexcept = default;

  explicit FactorGraph(const MarkovNetwork<Arg, AP, FP>& mn)
    requires (!std::is_void_v<AP> && !std::is_void_v<FP>)
    : FactorGraph() {
    for (auto* v : mn.vertices()) {
      add_argument(mn.argument(v), mn[v]);
    }
    for (auto e : mn.edges()) {
      add_factor(mn.domain(e), mn[e]);
    }
  }

  template <typename AP2, typename FP2, typename Converter>
  explicit FactorGraph(const MarkovNetwork<Arg, AP2, FP2>& mn, Converter converter)
    : FactorGraph() {
    for (auto* v : mn.vertices()) {
      if constexpr (!std::is_void_v<AP>) {
        add_argument(mn.argument(v), converter(mn[v]));
      } else {
        add_argument(mn.argument(v));
      }
    }
    for (auto e : mn.edges()) {
      if constexpr (!std::is_void_v<FP>) {
        add_factor(mn.domain(e), converter(mn[e]));
      } else {
        add_factor(mn.domain(e));
      }
    }
  }

  std::ranges::subrange<argument_iterator> arguments() const {
    return {arguments_.begin(), arguments_.end()};
  }

  std::ranges::subrange<factor_iterator> factors() const {
    return BipartiteGraph::vertices2();
  }

  const IntrusiveList<Factor>& factors(Arg u) const {
    return BipartiteGraph::neighbors(vertex(u));
  }

  const domain_type& arguments(Factor* u) const {
    return factor_annotation(u).value;
  }

  bool contains(Arg u) const {
    return arguments_.contains(u);
  }

  bool contains(Arg u, Factor* v) const {
    return contains(u) && BipartiteGraph::contains(v) && arguments(v).contains(u);
  }

  size_t degree(Arg u) const {
    return BipartiteGraph::degree(vertex(u));
  }

  size_t num_arguments() const {
    return arguments_.size();
  }

  size_t num_factors() const {
    return BipartiteGraph::num_vertices2();
  }

  Argument* vertex(Arg u) const {
    return arguments_.at(u);
  }

  Arg argument(Argument* u) const {
    return argument_annotation(u).value;
  }

  argument_property_reference operator[](Arg u) {
    return argument_annotation(vertex(u)).property();
  }

  const_argument_property_reference operator[](Arg u) const {
    return argument_annotation(vertex(u)).property();
  }

  argument_property_reference operator[](Argument* u) {
    return argument_annotation(u).property();
  }

  const_argument_property_reference operator[](Argument* u) const {
    return argument_annotation(u).property();
  }

  factor_property_reference operator[](Factor* f) {
    return factor_annotation(f).property();
  }

  const_factor_property_reference operator[](Factor* f) const {
    return factor_annotation(f).property();
  }

  markov_structure_type markov_graph() const {
    markov_structure_type mg;
    compute_vertex1_indices();
    for (Argument* u : vertices1()) {
      mg.add_vertex(argument(u));
    }
    for (Factor* f : factors()) {
      mg.add_clique(indices(f));
    }
    return mg;
  }

  Argument* add_argument(Arg u) {
    if (contains(u)) {
      throw std::invalid_argument("FactorGraph::add_argument: argument already exists");
    }
    Argument* v = BipartiteGraph::add_vertex1();
    argument_annotation(v).value = u;
    arguments_.emplace(u, v);
    return v;
  }

  template <typename T = AP>
  Argument* add_argument(Arg u, T property) requires (!std::is_void_v<T>) {
    Argument* v = add_argument(u);
    (*this)[v] = std::move(property);
    return v;
  }

  Factor* add_factor(domain_type args) {
    assert(args.is_sorted());
    std::vector<Argument*> neighbors;
    neighbors.reserve(args.size());
    for (const Arg& u : args) {
      neighbors.push_back(vertex(u));
    }
    Factor* f = BipartiteGraph::add_vertex2(std::move(neighbors));
    factor_annotation(f).value = std::move(args);
    return f;
  }

  template <typename T = FP>
  Factor* add_factor(domain_type args, T property) requires (!std::is_void_v<T>) {
    Factor* f = add_factor(std::move(args));
    (*this)[f] = std::move(property);
    return f;
  }

  void remove_argument(Arg u) {
    auto it = arguments_.find(u);
    assert(it != arguments_.end());
    BipartiteGraph::remove_vertex1(it->second);
    arguments_.erase(it);
  }

  void remove_factor(Factor* u) {
    BipartiteGraph::remove_vertex2(u);
  }

  void clear() {
    BipartiteGraph::clear();
    arguments_.clear();
  }

  template <typename Archive>
  void save(Archive& ar) const {
    ar(cereal::base_class<const BipartiteGraph>(this));

    ar(cereal::make_size_tag(num_arguments()));
    for (Argument* u : vertices1()) {
      ar(cereal::make_nvp("argument", argument_annotation(u).value));
      if constexpr (!std::is_void_v<AP>) {
        ar(cereal::make_nvp("property", operator[](u)));
      }
    }

    ar(cereal::make_size_tag(num_factors()));
    for (Factor* f : factors()) {
      if constexpr (!std::is_void_v<FP>) {
        ar(cereal::make_nvp("property", operator[](f)));
      }
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(cereal::base_class<BipartiteGraph>(this));

    cereal::size_type argument_count;
    ar(cereal::make_size_tag(argument_count));
    assert(argument_count == num_arguments());
    arguments_.clear();
    arguments_.reserve(argument_count);
    for (Argument* u : vertices1()) {
      Arg arg;
      ar(CEREAL_NVP(arg));
      argument_annotation(u).value = arg;
      arguments_.emplace(arg, u);
      if constexpr (!std::is_void_v<AP>) {
        ar(cereal::make_nvp("property", operator[](u)));
      }
    }

    cereal::size_type factor_count;
    ar(cereal::make_size_tag(factor_count));
    assert(factor_count == num_factors());
    for (Factor* f : factors()) {
      domain_type domain;
      domain.reserve(BipartiteGraph::neighbors(f).size());
      for (Argument* u : BipartiteGraph::neighbors(f)) {
        domain.push_back(argument(u));
      }
      factor_annotation(f).value = std::move(domain);
      if constexpr (!std::is_void_v<FP>) {
        ar(cereal::make_nvp("property", operator[](f)));
      }
    }
  }

private:
  ArgumentAnnotation& argument_annotation(Argument* u) {
    return opaque_cast<ArgumentAnnotation>(BipartiteGraph::property(u));
  }

  const ArgumentAnnotation& argument_annotation(Argument* u) const {
    return opaque_cast<ArgumentAnnotation>(BipartiteGraph::property(u));
  }

  FactorAnnotation& factor_annotation(Factor* f) {
    return opaque_cast<FactorAnnotation>(BipartiteGraph::property(f));
  }

  const FactorAnnotation& factor_annotation(Factor* f) const {
    return opaque_cast<FactorAnnotation>(BipartiteGraph::property(f));
  }

  void rebuild_map() {
    arguments_.clear();
    arguments_.reserve(num_vertices1());
    for (Argument* u : vertices1()) {
      arguments_.emplace(argument(u), u);
    }
  }

  ArgumentMap arguments_;
};

} // namespace libgm
