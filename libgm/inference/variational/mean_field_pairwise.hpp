#ifndef LIBGM_MEAN_FIELD_PAIRWISE_HPP
#define LIBGM_MEAN_FIELD_PAIRWISE_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/model/pairwise_markov_network.hpp>
#include <libgm/traits/pairwise_compatible.hpp>

#include <functional>
#include <unordered_map>

namespace libgm {

  /**
   * A class that runs the mean field algorithm for a pairwise Markov
   * network. The computation is performed sequentially in the order
   * of the vertices in Markov network.
   *
   * \tparam NodeF
   *         A factor type associated with vertices, typically in the
   *         canonical representation of the distribution, e.g., parray1.
   * \tparam EdgeF
   *         A factor type assocaited with edges, typically in the
   *         canonical representtaion of the distribution, e.g., parray2.
   *         This type must support the exp_log_multiply operation and
   *         must have the same argument and result type as NodeF.
   */
  template <typename NodeF, typename EdgeF = NodeF>
  class mean_field_pairwise {
    static_assert(pairwise_compatible<NodeF, EdgeF>::value,
                  "The node and edge factors are not pairwise compatible");

    // Public types
    //==========================================================================
  public:
    // Factorized Inference types
    typedef typename NodeF::real_type        real_type;
    typedef typename NodeF::result_type      result_type;
    typedef typename NodeF::argument_type    argument_type;
    typedef typename NodeF::assignment_type  assignment_type;
    typedef typename NodeF::probability_type belief_type;

    typedef pairwise_markov_network<NodeF, EdgeF> model_type;
    typedef typename model_type::vertex_type vertex_type;
    typedef typename model_type::edge_type edge_type;

    typedef typename argument_traits<argument_type>::hasher argument_hasher;

    // Public functions
    //==========================================================================
  public:
    /**
     * Creates a mean field engine for the given graph.
     * The graph vertices must not change after initialization
     * (the potentials may).
     */
    explicit mean_field_pairwise(const model_type* model)
      : model_(*model) {
      for (argument_type v : model_.vertices()) {
        beliefs_[v] = belief_type({v}, real_type(1));
      }
    }

    /**
     * Performs a single iteration of mean field.
     */
    real_type iterate() {
      real_type sum = 0.0;
      for (argument_type v : model_.vertices()) {
        sum += update(v);
      }
      return sum / model_.num_vertices();
    }

    /**
     * Returns the belief for a vertex.
     */
    const belief_type& belief(argument_type v) const {
      return beliefs_.at(v);
    }

  private:
    /**
     * Updates a single vertex.
     */
    real_type update(argument_type v) {
      NodeF result = model_[v];
      for (edge_type e : model_.in_edges(v)) {
        model_[e].exp_log_multiply(belief(e.source()), result);
      }
      result /= result.maximum();
      belief_type new_belief(result);
      new_belief.normalize();
      swap(beliefs_.at(v), new_belief);
      return sum_diff(new_belief, belief(v));
    }

    //! The underlying graphical model
    const model_type& model_;

    //! A map of current beliefs, one for each variable
    std::unordered_map<argument_type, belief_type, argument_hasher> beliefs_;

  }; // class mean_field_pairwise

} // namespace libgm

#endif
