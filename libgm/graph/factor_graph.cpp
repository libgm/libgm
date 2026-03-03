#include "factor_graph.hpp"

namespace libgm {

struct FactorGraph::Argument {
  /// The property associated with the argument.
  Object property;

  /// The list of factors, whose domain contains this argument.
  IntrusiveList<Factor> factors;

  /// The number of adjacent factors.
  size_t degree = 0;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(CEREAL_NVP(property));
  }

  Argument(Object property)
    : property(std::move(property)) {}
};

struct FactorGraph::Factor {
  /// The arguments associated with the factor.
  Domain arguments;

  /// The property associated with the factor
  Object property;

  /// The object owning this factor.
  Impl* impl;

  /// The hoook for all factors.
  IntrusiveList<Factor>::Hook hook;

  /// The hooks for the adjacency of arguments.
  IntrusiveList<Factor>::HookArray adjacency_hooks;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(CEREAL_NVP(arguments), CEREAL_NVP(property));
    if constexpr (ARCHIVE::is_loading::value) {
      adjacency_hooks.reset(arguments.size());
    }
  }

  Factor(Impl* impl)
    : impl(impl) {}

  Factor(Domain arguments, Object property, Impl* impl)
    : arguments(std::move(arguments)),
      property(std::move(property)),
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

  Impl() = default;

  template <typename ARCHIVE>
  void save(ARCHIVE& ar) const  {
    ar(CEREAL_NVP(arguments));

    // Save the factors as an array
    ar(cereal::make_size_tag(num_factors));
    for (Factor* factor : factors) {
      ar(*factor);
    }
  }

  template <typename ARCHIVE>
  void load(ARCHIVE& ar) {
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

const Object& FactorGraph::operator[](Arg u) const {
  return argument(u).property;
}

const Object& FactorGraph::operator[](Factor* u) const {
  return u->property;
}

Object& FactorGraph::operator[](Arg u) {
  return argument(u).property;
}

Object& FactorGraph::operator[](Factor* u) {
  return u->property;
}

std::ostream& operator<<(std::ostream& out, const FactorGraph& g) {
  out << "Arguments" << std::endl;
  for (Arg arg : g.arguments()) {
    out << arg << ": " << g[arg] << std::endl;
  }
  out << "Factors" << std::endl;
  size_t i = 0;
  for (FactorGraph::Factor* f : g.factors()) {
    out << i++ << ": " << f->arguments << " " << g[f] << std::endl;
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

bool FactorGraph::add_argument(Arg u, Object property) {
  assert(u != Arg());
  if (contains(u)) {
    return false;
  } else {
    impl().arguments.emplace(u, new Argument(std::move(property)));
    return true;
  }
}

FactorGraph::Factor* FactorGraph::add_factor(Domain arguments, Object property) {
  // Insert the new factor
  Factor* factor = new Factor(std::move(arguments), std::move(property), &impl());
  impl().factors.push_back(factor, factor->hook);
  ++impl().num_factors;

  // Connect arguments to the new factor
  for (size_t i = 0; i < factor->arguments.size(); ++i) {
    Argument& a = this->argument(arguments[i]);
    a.factors.push_back(factor, factor->adjacency_hooks[i]);
    ++a.degree;
  }

  return factor;
}

void FactorGraph::remove_argument(Arg u) {
  auto it = impl().arguments.find(u);
  Argument* argument = it->second;
  assert(argument->factors.empty());
  delete argument;
  impl().arguments.erase(it);
}

void FactorGraph::remove_factor(Factor* u) {
  for (Arg arg : u->arguments) {
    --argument(arg).degree;
  }
  --impl().num_factors;
  delete u;
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
        csr.combine_in(combination, f->property, domain.dims(f->arguments));
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
