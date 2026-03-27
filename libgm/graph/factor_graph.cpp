#include "factor_graph.hpp"

#include <cassert>

namespace libgm {

struct FactorGraph::Argument {
  /// The list of factors, whose domain contains this argument.
  IntrusiveList<Factor> factors;

  /// The number of adjacent factors.
  size_t degree = 0;

  template <typename Archive>
  void serialize(Archive& ar) {
    (void)ar;
  }

  Argument() = default;
};

struct FactorGraph::Factor {
  /// The arguments associated with the factor.
  Domain arguments;

  /// The object owning this factor.
  Impl* impl;

  /// The hoook for all factors.
  IntrusiveList<Factor>::Hook hook;

  /// The hooks for the adjacency of arguments.
  IntrusiveList<Factor>::HookArray adjacency_hooks;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(arguments));
    if constexpr (Archive::is_loading::value) {
      adjacency_hooks.reset(arguments.size());
    }
  }

  Factor(Impl* impl)
    : impl(impl) {}

  Factor(Domain arguments, Impl* impl)
    : arguments(std::move(arguments)),
      impl(impl),
      adjacency_hooks(this->arguments.size()) {}
};

struct FactorGraph::Impl {
  /// The properties and neighbors of arguments.
  ArgumentMap arguments;

  /// The properties and neighbors of factors.
  IntrusiveList<Factor> factors;

  /// The total number of factors.
  size_t num_factors = 0;

  PropertyLayout argument_property_layout;
  PropertyLayout factor_property_layout;

  Impl() = default;
  Impl(PropertyLayout argument_layout, PropertyLayout factor_layout)
    : argument_property_layout(argument_layout),
      factor_property_layout(factor_layout) {}

  Argument* allocate_argument() const {
    return argument_property_layout.allocate<Argument>();
  }

  Argument* add_argument(Arg u) {
    auto [it, inserted] = arguments.emplace(u, nullptr);
    if (!inserted) {
      return nullptr;
    }
    it->second = allocate_argument();
    return it->second;
  }

  Factor* allocate_factor(Domain arguments, Impl* impl) const {
    return factor_property_layout.allocate<Factor>(std::move(arguments), impl);
  }

  void free_argument(Argument* argument) const {
    argument_property_layout.free(argument);
  }

  void free_factor(Factor* factor) const {
    factor_property_layout.free(factor);
  }

  Factor* add_factor(Domain arguments) {
    Factor* factor = allocate_factor(std::move(arguments), this);
    factors.push_back(factor, factor->hook);
    ++num_factors;

    for (size_t i = 0; i < factor->arguments.size(); ++i) {
      Argument& a = *this->arguments.at(factor->arguments[i]);
      a.factors.push_back(factor, factor->adjacency_hooks[i]);
      ++a.degree;
    }

    return factor;
  }

  void clear() {
    for (auto it = factors.begin(); it != factors.end();) {
      free_factor(*it++);
    }
    num_factors = 0;

    for (auto [_, argument] : arguments) {
      free_argument(argument);
    }
    arguments.clear();
  }

  ~Impl() {
    clear();
  }

  std::unique_ptr<Impl> clone() const {
    auto result = std::make_unique<Impl>(argument_property_layout, factor_property_layout);

    for (auto [u, src] : arguments) {
      Argument* dst = result->add_argument(u);
      assert(dst);
      argument_property_layout.destroy_and_copy_construct(dst, src);
    }

    for (Factor* src : factors) {
      Factor* dst = result->add_factor(src->arguments);
      factor_property_layout.destroy_and_copy_construct(dst, src);
    }

    return result;
  }

  template <typename Archive>
  void save(Archive& ar) const  {
    ar(cereal::make_size_tag(arguments.size()));
    for (auto [u, _] : arguments) {
      ar(u);
    }

    // Save the factors as an array
    ar(cereal::make_size_tag(num_factors));
    for (Factor* factor : factors) {
      ar(factor->arguments);
    }
  }

  template <typename Archive>
  void load(Archive& ar) {
    clear();

    // Load arguments
    cereal::size_type argument_count;
    ar(cereal::make_size_tag(argument_count));
    for (size_t i = 0; i < argument_count; ++i) {
      Arg u;
      ar(u);
      arguments.emplace(u, allocate_argument());
    }

    // Load the factors
    cereal::size_type factor_count;
    ar(cereal::make_size_tag(factor_count));
    num_factors = 0;
    for (size_t i = 0; i < factor_count; ++i) {
      Domain factor_arguments;
      ar(factor_arguments);
      add_factor(std::move(factor_arguments));
    }
  }
};

FactorGraph::FactorGraph()
  : impl_(std::make_unique<Impl>()) {}

FactorGraph::FactorGraph(PropertyLayout argument_layout, PropertyLayout factor_layout)
  : impl_(std::make_unique<Impl>(argument_layout, factor_layout)) {}

FactorGraph::FactorGraph(const FactorGraph& other)
  : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

FactorGraph::FactorGraph(FactorGraph&& other) noexcept = default;

FactorGraph& FactorGraph::operator=(const FactorGraph& other) {
  if (this != &other) {
    impl_ = other.impl_ ? other.impl_->clone() : nullptr;
  }
  return *this;
}

FactorGraph& FactorGraph::operator=(FactorGraph&& other) noexcept = default;

FactorGraph::~FactorGraph() = default;

FactorGraph::Impl& FactorGraph::impl() {
  return *impl_;
}

const FactorGraph::Impl& FactorGraph::impl() const {
  return *impl_;
}

FactorGraph::Argument& FactorGraph::argument(Arg u) {
  return *impl().arguments.at(u);
}

const FactorGraph::Argument& FactorGraph::argument(Arg u) const {
  return *impl().arguments.at(u);
}

void swap(FactorGraph& a, FactorGraph& b) {
  std::swap(a.impl_, b.impl_);
}

std::ranges::subrange<FactorGraph::argument_iterator> FactorGraph::arguments() const {
  return { impl().arguments.begin(), impl().arguments.end() };
}

std::ranges::subrange<FactorGraph::factor_iterator> FactorGraph::factors() const {
  return { impl().factors.begin(), impl().factors.end() };
}

const IntrusiveList<FactorGraph::Factor>& FactorGraph::factors(Arg u) const {
  return argument(u).factors;
}

std::ranges::subrange<FactorGraph::out_edge1_iterator> FactorGraph::out_edges(Arg u) const {
  const IntrusiveList<Factor>& adjacent = factors(u);
  return {out_edge1_iterator(adjacent.begin(), u), out_edge1_iterator(adjacent.end(), u)};
}

std::ranges::subrange<FactorGraph::in_edge1_iterator> FactorGraph::in_edges(Arg u) const {
  const IntrusiveList<Factor>& adjacent = factors(u);
  return {in_edge1_iterator(adjacent.begin(), u), in_edge1_iterator(adjacent.end(), u)};
}

const Domain& FactorGraph::arguments(Factor* u) const {
  return u->arguments;
}

std::ranges::subrange<FactorGraph::out_edge2_iterator> FactorGraph::out_edges(Factor* u) const {
  const Domain& adjacent = arguments(u);
  return {out_edge2_iterator(adjacent.begin(), u), out_edge2_iterator(adjacent.end(), u)};
}

std::ranges::subrange<FactorGraph::in_edge2_iterator> FactorGraph::in_edges(Factor* u) const {
  const Domain& adjacent = arguments(u);
  return {in_edge2_iterator(adjacent.begin(), u), in_edge2_iterator(adjacent.end(), u)};
}

bool FactorGraph::contains(Arg u) const {
  return impl().arguments.find(u) != impl().arguments.end();
}

bool FactorGraph::contains(Factor* u) const {
  return u->impl == &impl();
}

bool FactorGraph::contains(Arg u, Factor* v) const {
  return contains(v) && v->arguments.contains(u);
}

size_t FactorGraph::degree(Arg u) const {
  return argument(u).degree;
}

size_t FactorGraph::degree(Factor* u) const {
  return u->arguments.size();
}

bool FactorGraph::empty() const {
  return impl().arguments.empty() && impl().factors.empty();
}

size_t FactorGraph::num_arguments() const {
  return impl().arguments.size();
}

size_t FactorGraph::num_factors() const {
  return impl().num_factors;
}

OpaqueRef FactorGraph::property(Arg u) {
  return impl().argument_property_layout.get(impl().arguments.at(u));
}

OpaqueCref FactorGraph::property(Arg u) const {
  return impl().argument_property_layout.get(static_cast<const Argument*>(impl().arguments.at(u)));
}

OpaqueRef FactorGraph::property(Factor* u) {
  return impl().factor_property_layout.get(u);
}

OpaqueCref FactorGraph::property(Factor* u) const {
  return impl().factor_property_layout.get(static_cast<const Factor*>(u));
}

std::ostream& operator<<(std::ostream& out, const FactorGraph& g) {
  out << "Arguments" << std::endl;
  for (Arg arg : g.arguments()) {
    out << arg << std::endl;
  }
  out << "Factors" << std::endl;
  size_t i = 0;
  for (FactorGraph::Factor* f : g.factors()) {
    out << i++ << ": " << f->arguments << std::endl;
  }
  return out;
}

MarkovNetwork FactorGraph::markov_network() const {
  MarkovNetwork mn;
  for (Factor* f : factors()) {
    mn.add_clique(f->arguments);
  }
  return mn;
}

bool FactorGraph::add_argument(Arg u) {
  assert(u != Arg());
  return impl().add_argument(u) != nullptr;
}

FactorGraph::Factor* FactorGraph::add_factor(Domain arguments) {
  assert(arguments.is_sorted());
  return impl().add_factor(std::move(arguments));
}

void FactorGraph::remove_argument(Arg u) {
  auto it = impl().arguments.find(u);
  assert(it != impl().arguments.end());
  Argument* argument = it->second;
  while (!argument->factors.empty()) {
    remove_factor(argument->factors.front());
  }
  impl().free_argument(argument);
  impl().arguments.erase(it);
}

void FactorGraph::remove_factor(Factor* u) {
  for (Arg arg : u->arguments) {
    --argument(arg).degree;
  }
  --impl().num_factors;
  impl().free_factor(u);
}

void FactorGraph::clear() {
  impl().clear();
}

} // namespace libgm
