#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/model/factor_graph.hpp>

namespace libgm {

template <typename F>
class VariableElimination {
public:
  using Factor = typename FactorGraph<F, F>::Factor;

  /// Constructs a variable elimination algorithm.
  VariableElimination(ShapeMap shape_map, const EliminationStrategy& strategy, const CommutativeSemiring<F>& csr)
    : shape_map_(std::move(shape_map)), strategy_(strategy), csr_(csr) {}

  /// Eliminates all variables other than the specified ones.
  void eliminate(FactorGraph<F, F>& fg, const Domain& retain) {
    MarkovStructure mg = fg.markov_graph();
    mg.eliminate(strategy_, [&](size_t v) {
      Arg arg = mg.argument(v);
      if (!retain.contains(arg)) {
        // Determine the union of all adjacent factor domains.
        Domain domain;
        for (Factor* f : fg.factors(arg)) {
          domain.append(fg.arguments(f));
        }
        domain.unique();

        // Combine all factors that have this variable as an argument
        F combination = csr_.init(domain.shape(shape_map_));
        for (Factor* f : fg.factors(arg)) {
          csr_.combine_in(combination, fg[f], domain.dims(fg.arguments(f)));
        }

        // Combine in the argument factor.
        csr_.combine_in(combination, fg[arg], domain.dims({arg}));

        // Delete the eliminated argument and the associated factors.
        fg.remove_argument(arg);

        // Add the new factor.
        F result = csr_.collapse(combination, domain.dims_omit(arg));
        domain.erase(arg);
        fg.add_factor(std::move(domain), std::move(result));
      }
    });
  }

  /// Joins all the factors in the product, with the given arguments, which must be a superset of fg.arguments().
  F join(FactorGraph<F, F>& fg, const Domain& domain) {
    F result = csr_.init(domain.shape(shape_map_));
    for (Arg arg : fg.arguments()) {
      csr_.combine_in(result, fg[arg], domain.dims({arg}));
    }
    for (Factor* f : fg.factors()) {
      csr_.combine_in(result, fg[f], domain.dims(fg.arguments(f)));
    }
    return result;
  }

  // Eliminates all variables other than the specified ones, and joins the rest.
  F eliminate_join(FactorGraph<F, F>& fg, const Domain& retain) {
    eliminate(fg, retain);
    return join(fg, retain);
  }

private:
  ShapeMap shape_map_;
  const EliminationStrategy& strategy_;
  const CommutativeSemiring<F>& csr_;
};

}
