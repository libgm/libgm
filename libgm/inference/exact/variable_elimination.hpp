#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/factor/utility/annotated.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
#include <libgm/graph/elimination_strategy.hpp>
#include <libgm/model/factor_graph.hpp>

namespace libgm {

/**
 * Eliminates variables from a factor graph using a chosen commutative semiring.
 *
 * \ingroup inference
 */
template <Argument Arg, typename F>
class VariableElimination {
public:
  using graph_type = FactorGraph<Arg, F, F>;
  using Factor = typename graph_type::Factor;

  VariableElimination(ShapeMap<Arg> shape_map, const EliminationStrategy& strategy, const CommutativeSemiring<F>& csr)
    : shape_map_(std::move(shape_map)), strategy_(strategy), csr_(csr) {}

  void eliminate(graph_type& fg, const Domain<Arg>& retain) {
    MarkovStructure<Arg> mg = fg.markov_graph();
    mg.eliminate(strategy_, [&](size_t v) {
      Arg arg = mg.argument(v);
      if (!retain.contains(arg)) {
        Domain<Arg> domain;
        for (Factor* f : fg.factors(arg)) {
          domain.append(fg.arguments(f));
        }
        domain.unique();

        F combination = csr_.init(domain.shape(shape_map_));
        for (Factor* f : fg.factors(arg)) {
          csr_.combine_in(combination, fg[f], domain.dims(fg.arguments(f)));
        }

        csr_.combine_in(combination, fg[arg], domain.dims({arg}));
        fg.remove_argument(arg);

        F result = csr_.collapse(combination, domain.dims_omit(arg));
        domain.erase(arg);
        fg.add_factor(std::move(domain), std::move(result));
      }
    });
  }

  F join(graph_type& fg, const Domain<Arg>& domain) {
    F result = csr_.init(domain.shape(shape_map_));
    for (Arg arg : fg.arguments()) {
      csr_.combine_in(result, fg[arg], domain.dims({arg}));
    }
    for (Factor* f : fg.factors()) {
      csr_.combine_in(result, fg[f], domain.dims(fg.arguments(f)));
    }
    return result;
  }

  F eliminate_join(graph_type& fg, const Domain<Arg>& retain) {
    eliminate(fg, retain);
    return join(fg, retain);
  }

private:
  ShapeMap<Arg> shape_map_;
  const EliminationStrategy& strategy_;
  const CommutativeSemiring<F>& csr_;
};

} // namespace libgm
