#ifndef LIBGM_VARIABLE_ELIMINATION_HPP
#define LIBGM_VARIABLE_ELIMINATION_HPP

#include <libgm/factor/util/commutative_semiring.hpp>
#include <libgm/graph/algorithm/eliminate.hpp>
#include <libgm/graph/undirected_graph.hpp>

#include <list>

namespace libgm {

  /**
   * The variable elimination algorithm.
   * Given a collection of factors and a subset of their arguments,
   * this method efficiently combines the factors and collapses the result
   * to the desired arguments.
   *
   * \tparam F A type representing the factor
   * \tparam Strategy A type that model EliminationStrategy concept
   *
   * \param factors
   *        The collection of factors, modified in place.
   * \param retain
   *        The retained arguments.
   * \param csr
   *        An object (such as sum_product) that determines how factors are
   *        combined and collapse.
   * \param elim_strategy
   *        The strategy that determines the order in which variable are
   *         eliminated.
   *
   * \ingroup inference
   */
  template <typename F, typename Strategy = min_degree_strategy>
  void variable_elimination(std::list<F>& factors,
                            const typename F::domain_type& retain,
                            const commutative_semiring<F>& csr,
                            Strategy strategy = Strategy()) {
    typedef typename F::argument_type argument_type;

    // construct the Markov graph for the input factors
    undirected_graph<argument_type> graph;
    for (const F& factor : factors) {
      make_clique(graph, factor.arguments());
    }

    // eliminate variables
    eliminate(graph, [&](argument_type v) {
        if (!retain.count(v)) {
          // Combine all factors that have this variable as an argument
          F combination = csr.combine_init();
          for (auto it = factors.begin(); it != factors.end();) {
            if (it->arguments().count(v)) {
              if (superset(combination.arguments(), it->arguments())) {
                csr.combine_in(combination, *it);
              } else {
                combination = csr.combine(combination, *it);
              }
              factors.erase(it++);
            } else {
              ++it;
            }
          }
          factors.push_back(csr.collapse_out(combination, {v}));
        }
      }, strategy);
  }

} // namespace libgm

#endif
