#pragma once

#include <ankerl/unordered_dense.h>

#include <libgm/graph/factor_graph.hpp>

namespace libgm {

/**
 * A class that runs the mean field algorithm for a factor graph.
 * The computation is performed synchronously, first for all
 * arguments and then for all factors. The number of worker
 * threads is controlled by a parameter to the constructor.
 */
template <typename ArgumentF, typename FactorF>
class MeanField {
public:
  using real_type = typename ArgumentF::real_type;
  using belief_type = typename ArgumentF::probability_type;

  using graph_type = FactorGraphT<ArgumentF, FactorF>;
  using edge_descriptor = typename graph_type::fa_edge_descriptor;
  using Factor = typename graph_type::Factor;

  /// Creates a mean field engine for the given graph.
  /// The graph vertices must not change after initialization (the potentials may).
  /// \param num_threads the number of worker threads
  explicit MeanField(const graph_type& graph, ShapeMap shape_map, size_t nthreads = 1)
    : graph_(graph), nthreads_(nthreads) {
    for (Arg arg : graph_.arguments()) {
      size_t shape = shape_map(arg);
      belief_type belief(shape, real_type(1));
      belief.normalize();
      beliefs_.emplace(arg, std::move(belief));
      for (edge_descriptor e : graph_.in_edges(arg)) {
        messages_.emplace(e, ArgumentF(shape));
      }
    }
  }

  /// Performs a single iteration of mean field.
  real_type iterate() {
    for (auto& [edge, message] : messages_) {
      update_message(edge, message);
    }

    real_type diff = real_type(0);
    for (auto& [arg, belief] : beliefs_) {
      diff += update_belief(arg, belief);
    }
    return diff / graph_.num_arguments();
  }

  /// Returns the belief for an argument.
  const belief_type& belief(Arg argument) const {
    return beliefs_.at(argument);
  }

private:
  void update_message(edge_descriptor e, ArgumentF& result) const {
    const Domain& domain = graph_.arguments(e.source());
    std::vector<belief_type> beliefs;
    for (Arg arg : domain) {
      if (arg != e.target()) {
        beliefs.push_back(beliefs_.at(arg));
      }
    }

    result = graph_[e.source()].expected_log_dim(beliefs, domain.index(e.target()));
  }

  real_type update_belief(Arg arg, belief_type& belief) const {
    ArgumentF result = graph_[arg];
    for (edge_descriptor e : graph_.in_edges(arg)) {
      result *= messages_.at(e);
    }
    result /= result.maximum();
    belief_type new_belief = result.probability();
    new_belief.normalize();
    real_type diff = max_diff(belief, new_belief);
    belief = std::move(new_belief);
    return diff;
  }

  /// The underlying graphical model.
  const graph_type& graph_;

  /// The number of worker threads.
  size_t nthreads_;

  /// A map of current beliefs for arguments.
  ankerl::unordered_dense::map<Arg, belief_type> beliefs_;

  /// The message for each edge.
  ankerl::unordered_dense::map<edge_descriptor, ArgumentF> messages_;
};

}
