#include "factor_graph.hpp"

#include <cassert>
#include <stdexcept>

namespace libgm {

struct FactorGraph::Argument {
  /// The list of factors, whose domain contains this argument.
  IntrusiveList<Factor> factors;

  /// The number of adjacent factors.
  size_t degree = 0;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
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

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(CEREAL_NVP(arguments));
    if constexpr (ARCHIVE::is_loading::value) {
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

struct FactorGraph::Impl : Object::Impl {
  /// The properties and neighbors of arguments.
  ArgumentMap arguments;

  /// The properties and neighbors of factors.
  IntrusiveList<Factor> factors;

  /// The total number of factors.
  size_t num_factors = 0;

  PropertyLayout argument_property_layout;
  PropertyLayout factor_property_layout;
  size_t argument_property_offset = sizeof(Argument);
  size_t argument_allocation_size = sizeof(Argument);
  size_t factor_property_offset = sizeof(Factor);
  size_t factor_allocation_size = sizeof(Factor);

  Impl() = default;
  Impl(PropertyLayout argument_layout, PropertyLayout factor_layout)
    : argument_property_layout(argument_layout),
      factor_property_layout(factor_layout) {
    argument_property_offset = argument_property_layout.align_up(sizeof(Argument));
    argument_allocation_size = argument_property_offset + argument_property_layout.size;
    factor_property_offset = factor_property_layout.align_up(sizeof(Factor));
    factor_allocation_size = factor_property_offset + factor_property_layout.size;
  }

  void* argument_property(Argument* argument) const {
    return reinterpret_cast<char*>(argument) + argument_property_offset;
  }

  const void* argument_property(const Argument* argument) const {
    return reinterpret_cast<const char*>(argument) + argument_property_offset;
  }

  void* factor_property(Factor* factor) const {
    return reinterpret_cast<char*>(factor) + factor_property_offset;
  }

  const void* factor_property(const Factor* factor) const {
    return reinterpret_cast<const char*>(factor) + factor_property_offset;
  }

  Argument* allocate_argument() const {
    void* buffer = ::operator new(argument_allocation_size);
    Argument* argument = new (buffer) Argument();
    if (argument_property_layout.size != 0) {
      assert(argument_property_layout.default_constructor);
      argument_property_layout.default_constructor(argument_property(argument));
    }
    return argument;
  }

  Factor* allocate_factor(Domain arguments, Impl* impl) const {
    void* buffer = ::operator new(factor_allocation_size);
    Factor* factor = new (buffer) Factor(std::move(arguments), impl);
    if (factor_property_layout.size != 0) {
      assert(factor_property_layout.default_constructor);
      factor_property_layout.default_constructor(factor_property(factor));
    }
    return factor;
  }

  void free_argument(Argument* argument) const {
    if (argument_property_layout.size != 0) {
      assert(argument_property_layout.deleter);
      argument_property_layout.deleter(argument_property(argument));
    }
    argument->~Argument();
    ::operator delete(argument);
  }

  void free_factor(Factor* factor) const {
    if (factor_property_layout.size != 0) {
      assert(factor_property_layout.deleter);
      factor_property_layout.deleter(factor_property(factor));
    }
    factor->~Factor();
    ::operator delete(factor);
  }

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) const  {
    if (argument_property_layout.size != 0 || factor_property_layout.size != 0) {
      throw std::logic_error("Serializing FactorGraph properties is unsupported.");
    }
    ar(CEREAL_NVP(arguments));

    // Save the factors as an array
    ar(cereal::make_size_tag(num_factors));
    for (Factor* factor : factors) {
      ar(*factor);
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
    if (argument_property_layout.size != 0 || factor_property_layout.size != 0) {
      throw std::logic_error("Deserializing FactorGraph properties is unsupported.");
    }
    ar(CEREAL_NVP(arguments));

    // Load the factors
    cereal::size_type size;
    ar(cereal::make_size_tag(size));
    num_factors = size;
    for (size_t i = 0; i < num_factors; ++i) {
      Factor* factor = new Factor(this);
      ar(*factor);
      factors.push_back(factor, factor->hook);
      for (size_t i = 0; i < factor->arguments.size(); ++i) {
        Argument& a = *arguments.at(factor->arguments[i]);
        a.factors.push_back(factor, factor->adjacency_hooks[i]);
        ++a.degree;
      }
    }
  }
};

FactorGraph::FactorGraph()
  : Object(std::make_unique<Impl>()) {}

FactorGraph::FactorGraph(PropertyLayout argument_layout, PropertyLayout factor_layout)
  : Object(std::make_unique<Impl>(argument_layout, factor_layout)) {}

FactorGraph::Impl& FactorGraph::impl() {
  return static_cast<Impl&>(*impl_);
}

const FactorGraph::Impl& FactorGraph::impl() const {
  return static_cast<const Impl&>(*impl_);
}

FactorGraph::Argument& FactorGraph::argument(Arg u) {
  return *impl().arguments.at(u);
}

const FactorGraph::Argument& FactorGraph::argument(Arg u) const {
  return *impl().arguments.at(u);
}

void swap(FactorGraph& a, FactorGraph& b) {
  swap(a.impl_, b.impl_);
}

SubRange<FactorGraph::argument_iterator> FactorGraph::arguments() const {
  return { impl().arguments.begin(), impl().arguments.end() };
}

SubRange<FactorGraph::factor_iterator> FactorGraph::factors() const {
  return { impl().factors.begin(), impl().factors.end() };
}

const Domain& FactorGraph::arguments(Factor* u) const {
  return u->arguments;
}

const IntrusiveList<FactorGraph::Factor>& FactorGraph::factors(Arg u) const {
  return argument(u).factors;
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

void* FactorGraph::property(Arg u) {
  return impl().argument_property(impl().arguments.at(u));
}

const void* FactorGraph::property(Arg u) const {
  return impl().argument_property(impl().arguments.at(u));
}

void* FactorGraph::property(Factor* u) {
  return impl().factor_property(u);
}

const void* FactorGraph::property(Factor* u) const {
  return impl().factor_property(u);
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

MarkovNetworkT<> FactorGraph::markov_network() const {
  MarkovNetworkT<> mn;
  for (Factor* f : factors()) {
    mn.add_clique(f->arguments);
  }
  return mn;
}

bool FactorGraph::add_argument(Arg u) {
  assert(u != Arg());
  if (contains(u)) {
    return false;
  } else {
    impl().arguments.emplace(u, impl().allocate_argument());
    return true;
  }
}

FactorGraph::Factor* FactorGraph::add_factor(Domain arguments) {
  // Insert the new factor
  Factor* factor = impl().allocate_factor(std::move(arguments), &impl());
  impl().factors.push_back(factor, factor->hook);
  ++impl().num_factors;

  // Connect arguments to the new factor
  for (size_t i = 0; i < factor->arguments.size(); ++i) {
    Argument& a = this->argument(factor->arguments[i]);
    a.factors.push_back(factor, factor->adjacency_hooks[i]);
    ++a.degree;
  }

  return factor;
}

void FactorGraph::remove_argument(Arg u) {
  auto it = impl().arguments.find(u);
  Argument* argument = it->second;
  assert(argument->factors.empty());
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

void FactorGraph::eliminate(const Domain& retain,
                            const CommutativeSemiring& csr,
                            const ShapeMap& shape_map,
                            const EliminationStrategy& strategy) {
  MarkovNetworkT<> mn = markov_network();
  mn.eliminate(strategy, [&](Arg arg) {
    if (!retain.contains(arg)) {
      // Determine the union of all adjacent factor domains.
      Domain domain;
      for (Factor* f : factors(arg)) {
        domain.append(f->arguments);
      }
      domain.unique();

      // Combine all factors that have this variable as an argument
      Object combination = csr.init(domain.shape(shape_map));
      for (Factor* f : factors(arg)) {
        csr.combine_in(combination,
                       *static_cast<Object*>(property(f)),
                       domain.dims(f->arguments));
      }

      // Delete the eliminated argument and the associated factors.
      remove_argument(arg);

      // Add the new factor.
      Object result = csr.collapse(combination, domain.dims_omit(arg));
      domain.erase(arg);
      add_factor(std::move(domain), std::move(result));
    }
  });
}

} // namespace libgm
