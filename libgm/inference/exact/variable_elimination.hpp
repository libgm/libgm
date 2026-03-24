#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/graph/factor_graph.hpp>

namespace libgm {

template <typename F>
struct VariableElimination {
  // Eliminates all variables other than the specified ones.
  template <typename T>
  void eliminate(FactorGraphT<T, F>& fg, const Domain& retain, const ShapeMap& shape_map, const EliminationStrategy& strategy,
                 const CommutativeSemiring<F>& csr) {
    using Factor = FactorGraph::Factor;

    MarkovNetwork mn = fg.markov_network();
    mn.eliminate(strategy, [&](Arg arg) {
      if (!retain.contains(arg)) {
        // Determine the union of all adjacent factor domains.
        Domain domain;
        for (Factor* f : fg.factors(arg)) {
          domain.append(fg.arguments(f));
        }
        domain.unique();

        // Combine all factors that have this variable as an argument
        F combination = csr.init(domain.shape(shape_map));
        for (Factor* f : fg.factors(arg)) {
          csr.combine_in(combination, fg[f], domain.dims(fg.arguments(f)));
        }

        // Delete the eliminated argument and the associated factors.
        fg.remove_argument(arg);

        // Add the new factor.
        F result = csr.collapse(combination, domain.dims_omit(arg));
        domain.erase(arg);
        fg.add_factor(std::move(domain), std::move(result));
      }
    });
  }

  // Combines all factors with the given commutative semiring.
  template <typename T>
  Annotated<F> combine_all(const FactorGraphT<T, F>& fg, const ShapeMap& shape_map, const CommutativeSemiring<F>& csr) {
    Domain domain(fg.arguments());
    domain.sort();
    F result = csr.init(domain.shape(shape_map));
    for (FactorGraph::Factor* f : fg.factors()) {
      csr.combine_in(result, fg[f], domain.dims(fg.arguments(f)));
    }
    return {std::move(result), std::move(domain)};
  }
};

}
