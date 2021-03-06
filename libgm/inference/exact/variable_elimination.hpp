#ifndef LIBGM_VARIABLE_ELIMINATION_HPP
#define LIBGM_VARIABLE_ELIMINATION_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/annotated.hpp>
#include <libgm/factor/utility/commutative_semiring.hpp>
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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type representing the factor
   * \tparam Strategy
   *         A type that model EliminationStrategy concept
   *
   * \param factors
   *        The collection of annotated factors, modified in place.
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
  template <typename Arg, typename F, typename Strategy = min_degree_strategy>
  void variable_elimination(std::list<annotated<Arg, F> >& factors,
                            const domain<Arg>& retain,
                            const commutative_semiring<Arg, F>& csr,
                            Strategy strategy = Strategy()) {

    // construct the Markov graph for the input factors
    undirected_graph<Arg> graph;
    for (const annotated<Arg, F>& factor : factors) {
      graph.make_clique(factor.domain);
    }

    // eliminate variables
    eliminate(graph, [&](Arg arg) {
        if (!retain.count(arg)) {
          // Combine all factors that have this variable as an argument
          annotated<Arg, F> combination = csr.combine_init();
          for (auto it = factors.begin(); it != factors.end();) {
            if (it->domain.count(arg)) {
              csr.combine_in(combination, *it);
              factors.erase(it++);
            } else {
              ++it;
            }
          }
          factors.push_back(csr.eliminate(combination, arg));
        }
      }, strategy);
  }

} // namespace libgm

#endif
