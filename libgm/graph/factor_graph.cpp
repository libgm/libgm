#include "factor_graph.hpp"

namespace libgm {

struct FactorGraph::Argument {
  Object property;
  FactorSet factors;
};

struct FactorGraph::Factor : VertexBase, Domain {
  Factor(Domain cluster, Object property)
    : Domain(std::move(cluster)), property(std::move(property)) {}

  const FactorGraph* graph;
  Object property;
  size_t index;
};

struct FactorGraph::Impl : Object::Impl {
  /// The properties and neighbors of type-1 vertices.
  ArgumentMap arguments;

  /// The properties and neighbors of type-2 vertices.
  FactorList factors;
};

friend void swap(FactorGraph& a, FactorGraph& b) {
  swap(a.impl_, b.impl_);
}

boost::iterator_range<vertex1_iterator> FactorGraph::arguments() const {
  return { impl().arguments.begin(), impl().arguments.end() };
}

boost::iterator_range<vertex2_iterator> FactorGraph::factors() const {
  return { impl().factors.begin(), impl().factors.end() };
}

const Domain& FactorGraph::arguments(Factor* u) const {
  return *u;
}

const FactorGraph::FactorSet& FactorGraph::factors(Arg u) const {
  return argument(u).factors;
}

bool FactorGraph::contains(Arg u) const {
  return impl().arguments.find(u) != impl().arguments.end();
}

bool FactorGraph::contains(Factor* u) const {
  return u->graph == this;
}

bool FactorGraph::contains(Arg u, Factor* v) const {
  return contains(v) && v->contains(u);
}

size_t FactorGraph::degree(Arg u) const {
  return argument(u).factors.size();
}

size_t FactorGraph::degree(Factor* u) const {
  return u->size();
}

bool FactorGraph::empty() const {
  return impl().arguments.empty() && impl().factors.empty();
}

size_t FactorGraph::num_arguments() const {
  return impl().arguments.size();
}

size_t FactorGraph::num_factors() const {
  return impl().factors.size();
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

friend std::ostream& operator<<(std::ostream& out, const FactorGraph& g) {
  out << "Arguments" << std::endl;
  for (Arg arg : g.arguments()) {
    out << arg << ": " << g[arg] << std::endl;
  }
  out << "Factors" << std::endl;
  for (Factor* f : g.factors()) {
    out << f->id << ": " << f->cluster() << g[f] << std::endl;
  }
  return out;
}

MarkovGraphT<> FactorGraph::markov_graph() const {
  MarkovGraphT<> mn;
  for (Factor* f : factors()) {
    mn.add_clique(*f);
  }
  return mn;
}

bool FactorGraph::add_argument(Arg u, Object property) {
  assert(u != Arg());
  if (contains(u)) {
    return false;
  } else {
    argument(u).property = std::move(property);
    return true;
  }
}

Factor* FactorGraph::add_factor(Domain args, Object property) {
  // Insert the new factor
  Factor* factor = new Factor(std::move(args), std::move(property));
  impl().factors.push_back(*factor);

  // Connect arguments to the new factor
  for (Arg arg : args) {
    argument(arg).factors.insert(factor);
  }

  return factor;
}

void FactorGraph::remove_argument(Arg u) {
  auto it = impl().arguments.find(u);
  Argument* argument = *it;
  assert(argument->factors.empty());
  delete argument;
  impl().arguments.erase(it);
}

void FactorGraph::remove_factor(Factor* u) {
  for (Arg arg : *u) {
    argument(arg).factors.erase(u);
  }
  impl().factors.erase(u);
  delete u;
}

void FactorGraph::save(oarchive& ar) const {
  ar << num_arguments() << num_factors();
  for (auto [u, argument] : impl().arguments) {
    ar << u << argument->property;
  }
  for (Factor* factor : factors()) {
    ar << factor->id << factor->cluster() << factor->property;
  }
}

void FactorGraph::load(iarchive& ar) {
  clear();
  size_t num_arguments, num_factors;
  Arg arg;
  Factor* f;
  ar >> num_arguments >> num_factors;
  while (num_arguments-- > 0) {
    ar >> arg;
    ar >> argument(u).property;
  }
  while (num_factors-- > 0) {
    f = new Factor;
    ar >> f->id >> f->cluster() >> f->property;
    impl().factors.push_back(f);
    // TODO: edges from arguments
  }
}

void FactorGraph::eliminate(const Domain& retain, EliminationStrategy strategy) {
  MarkovGraphT<> mn = markov_graph();
  mn.eliminate([this, &csr](Arg arg) {
    if (!retain.contains(arg)) {
      // Determine the union of all adjacent factor domains.
      Domain domain;
      for (Factor* f : factors(arg)) {
        domain.append(*f);
      }
      domain.unique();

      // Combine all factors that have this variable as an argument
      Object combination = csr.init(domain.shape(shape_map));
      for (Factor* f : factors(arg)) {
        csr.combine_in(combination, f->property, domain.dims(*f));
      }

      // Delete the eliminated argument and the associated factors.
      remove_argument(arg);

      // Add the new factor.
      add_factor(std::move(domain), csr.eliminate(combination, domain.index(arg)));
    }
  }, strategy);
}

} // namespace libgm
