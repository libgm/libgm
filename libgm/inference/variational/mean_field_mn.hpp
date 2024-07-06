#ifndef LIBGM_MEAN_FIELD_PAIRWISE_HPP
#define LIBGM_MEAN_FIELD_PAIRWISE_HPP

#include <libgm/factor/utility/traits.hpp>
#include <libgm/argument/traits.hpp>
#include <libgm/model/pairwise_markov_network.hpp>

#include <functional>
#include <unordered_map>

namespace libgm {

/**
 * A class that runs the mean field algorithm for a pairwise Markov
 * network. The computation is performed sequentially in the order
 * of the vertices in Markov network.
 */
class MeanFieldPairwise {
public:
  // Factor types
  struct NodePotential {

  };

  struct EdgePotential {

  };

  struct Belief {

  };

  /**
   * Creates a mean field engine for the given graph.
   * The graph vertices must not change after initialization
   * (the potentials may).
   */
  explicit MeanFieldPairwise(const MarkovNetwork<NodePotential, EdgePotential>* graph)
    : graph_(*graph) {
    for (Arg v : graph_.vertices()) {
      beliefs_[v] = Belief::ones(v->shape({v}, shape_map), bt);
    }
  }

  /**
   * Performs a single iteration of mean field.
   */
  Real iterate() {
    Real sum = 0.0;
    for (Arg v : graph_.vertices()) {
      sum += update(v);
    }
    return sum / graph_.num_vertices();
  }

  /**
   * Returns the belief for a vertex.
   */
  const Belief& belief(Arg v) const {
    return beliefs_.at(v);
  }

private:
  /**
   * Updates a single vertex.
   */
  Real update(Arg v) {
    NodePotential result = graph_[v];
    for (edge_descriptor e : graph_.in_edges(v)) {
      if (e.forward()) {
        result.multiply_in(graph_[e].expected_log_head(belief(e.source()), et), nt);
      } else {
        result.multiply_in(graph_[e].expected_log_tail(belief(e.source()), et), nt);
      }
    }
    result.divide_in(result.max(), nt);
    Belief new_belief = result.probability(, nt);
    new_belief.normalize(bt);
    swap(beliefs_.at(v), new_belief);
    return sum_diff(new_belief, belief(v), bt);
  }

  /// The underlying graphical model
  const graph_type& graph_;

  /// A map of current beliefs, one for each variable
  ankerl::unordered_dense:::map<Arg, Belief> beliefs_;

}; // class MeanFIeldPairwise

/**
 *
 * \tparam NodeF
 *         A factor type associated with vertices, typically in the
 *         logarithmic representation of the distribution.
 * \tparam EdgeF
 *         A factor type associated with edges, typically in the
 *         logarithmic representation of the distribution.
 *         This type must support the exp_log_multiply operation and
 *         must have the same argument and result type as NodeF.
 */
template <typename NodeF, typename EdgeF>
class MeanFieldPairwiseT : public MeanFieldPairwise {
public:
  MeanFieldPairwiseT();
};

} // namespace libgm

#endif
