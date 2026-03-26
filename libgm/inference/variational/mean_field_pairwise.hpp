#pragma once

#include <ankerl/unordered_dense.h>

#include <libgm/argument/argument.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/graph/markov_network.hpp>

namespace libgm {

/**
 * A class that runs the mean field algorithm for a pairwise Markov network.
 * The computation is performed synchronously over all vertices; updates are
 * computed from the previous belief state and then committed together.
 *
 * \tparam NodeF
 *         A factor type associated with vertices, typically in the logarithmic
 *         representation of the distribution.
 * \tparam EdgeF
 *         A factor type associated with edges, typically in the logarithmic
 *         representation of the distribution. This type must support
 *         `expected_log_front` and `expected_log_back` with the belief type of
 *         `NodeF`.
 */
template <typename NodeF, typename EdgeF>
class MeanFieldPairwise {
public:
  using real_type = typename NodeF::real_type;
  using belief_type = typename NodeF::probability_type;

  /// Creates a mean field engine for the given graph.
  explicit MeanFieldPairwise(
      const MarkovNetworkT<NodeF, EdgeF>& graph,
      ShapeMap shape_map)
    : graph_(graph) {
    for (Arg v : graph_.vertices()) {
      belief_type belief(shape_map(v), real_type(1));
      belief.normalize();
      beliefs_.emplace(v, std::move(belief));
    }
  }

  /// Performs a single iteration of mean field.
  real_type iterate() {
    real_type sum = real_type(0);
    for (Arg v : graph_.vertices()) {
      sum += update(v, beliefs_.at(v));
    }
    return sum / graph_.num_vertices();
  }

  /// Returns the belief for a vertex.
  const belief_type& belief(Arg v) const {
    return beliefs_.at(v);
  }

private:
  /// Computes the next belief for a single vertex using the previous state.
  real_type update(Arg v, belief_type& belief) const {
    NodeF result = graph_[v];
    for (UndirectedEdge<Arg> e : graph_.in_edges(v)) {
      if (e.is_nominal()) {
        result *= graph_[e].expected_log_front(beliefs_.at(e.source()));
      } else {
        result *= graph_[e].expected_log_back(beliefs_.at(e.source()));
      }
    }
    result /= result.maximum();
    belief_type new_belief = result.probability();
    new_belief.normalize();
    real_type diff = max_diff(belief, new_belief);
    belief = std::move(new_belief);
    return diff;
  }

  /// The underlying graphical model.
  const MarkovNetworkT<NodeF, EdgeF>& graph_;

  /// A map of current beliefs, one for each variable.
  ankerl::unordered_dense::map<Arg, belief_type> beliefs_;
};

} // namespace libgm
