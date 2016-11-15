#ifndef LIBGM_MEAN_FIELD_PAIRWISE_HPP
#define LIBGM_MEAN_FIELD_PAIRWISE_HPP

#include <libgm/factor/traits.hpp>
#include <libgm/argument/argument_traits.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <functional>
#include <unordered_map>

namespace libgm {

  /**
   * A class that runs the mean field algorithm for a pairwise Markov
   * network. The computation is performed sequentially in the order
   * of the vertices in Markov network.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam NodeF
   *         A factor type associated with vertices, typically in the
   *         logarithmic representation of the distribution.
   * \tparam EdgeF
   *         A factor type associated with edges, typically in the
   *         logarithmic representation of the distribution.
   *         This type must support the exp_log_multiply operation and
   *         must have the same argument and result type as NodeF.
   */
  template <typename Arg, typename NodeF, typename EdgeF = NodeF>
  class mean_field_pairwise {
    static_assert(are_pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

    // Public types
    //--------------------------------------------------------------------------
  public:
    // Model types
    using graph_type      = pairwise_markov_network<Arg, NodeF, EdgeF>;
    using argument_hasher = typename argument_traits<Arg>::hasher;

    // Factor types
    using real_type   = typename NodeF::real_type;
    using result_type = typename NOdeF::result_type;
    using belief_type = typename NodeF::probability_type;

    // Public functions
    //--------------------------------------------------------------------------
  public:
    /**
     * Creates a mean field engine for the given graph.
     * The graph vertices must not change after initialization
     * (the potentials may).
     */
    explicit mean_field_pairwise(const graph_type* graph)
      : graph_(*graph) {
      for (Arg v : graph_.vertices()) {
        beliefs_[v] = belief_type(NodeF::shape(v), real_type(1));
      }
    }

    /**
     * Performs a single iteration of mean field.
     */
    real_type iterate() {
      real_type sum = 0.0;
      for (Arg v : graph_.vertices()) {
        sum += update(v);
      }
      return sum / graph_.num_vertices();
    }

    /**
     * Returns the belief for a vertex.
     */
    const belief_type& belief(Arg v) const {
      return beliefs_.at(v);
    }

  private:
    /**
     * Updates a single vertex.
     */
    real_type update(Arg v) {
      NodeF result = graph_[v];
      for (undirected_edge<Arg> e : graph_.in_edges(v)) {
        if (e.forward()) {
          result *= graph_[e].head().expected_log(belief(e.source()));
        } else {
          result *= graph_[e].tail().expected_log(belief(e.source()));
        }
      }
      result /= result.max();
      belief_type new_belief = result.probability();
      new_belief.normalize();
      swap(beliefs_.at(v), new_belief);
      return sum_diff(new_belief, belief(v));
    }

    //! The underlying graphical model
    const graph_type& graph_;

    //! A map of current beliefs, one for each variable
    std::unordered_map<Arg, belief_type, argument_hasher> beliefs_;

  }; // class mean_field_pairwise

} // namespace libgm

#endif
