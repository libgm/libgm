#pragma once

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/model/markov_network.hpp>
#include <libgm/inference/belief_defs.hpp>
#include <libgm/inference/loopy/pairwise_bp.hpp>

namespace libgm {

/**
 * Belief propagation schedule for the specific node factor type.
 */
template <typename F>
struct PairwiseBeliefSchedule {
  using real_type = typename F::real_type;

  virtual ~PairwiseBeliefSchedule() = default;

  /// Initializes all the messages using the given generator (must be provided).
  virtual void initialize(std::function<F(Arg)> gen) = 0;

  /// Initializes all the messages by initializing them to identity using the provided shape_map.
  virtual void initialize(const ShapeMap& shape_map) = 0;

  /// Perform one iteration.
  virtual real_type iterate() = 0;
};

/**
 * Loopy BP schedule that updates the messages synchronously.
 */
template <typename F>
class SynchronousPropagationSchedule : public PairwiseBeliefSchedule<F> {
public:
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<F>::edge_descriptor;

  explicit SynchronousPropagationSchedule(PairwiseBeliefState<F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
    state_.swap(messages_);
    state_.initialize(gen);
  }

  void initialize(const ShapeMap& shape_map) override {
    state_.initialize(shape_map);
    state_.swap(messages_);
    state_.initialize(shape_map);
  }

  real_type iterate() override {
    real_type residual(0);
    const UndirectedGraph& graph = state_.graph();
    for (auto v : graph.vertices()) {
      for (auto e : graph.in_edges(v)) {
        F& cur_message = messages_[e];
        F new_message = state_.compute_message(e);
        residual += diff_(cur_message, new_message);
        update_(cur_message, std::move(new_message));
      }
    }
    state_.swap(messages_);
    return residual / graph.num_edges() / 2;
  }

private:
  PairwiseBeliefState<F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
  ankerl::unordered_dense::map<edge_descriptor, F> messages_;
};

/**
 * Loopy BP schedule that updates the messages in a round-robin manner.
 */
template <typename F>
struct AsynchronousPropagationSchedule : PairwiseBeliefSchedule<F> {
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<F>::edge_descriptor;

  explicit AsynchronousPropagationSchedule(PairwiseBeliefState<F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
  }

  void initialize(const ShapeMap& shape_map) override {
    state_.initialize(shape_map);
  }

  real_type iterate() override {
    real_type residual(0);
    const UndirectedGraph& graph = state_.graph();
    for (auto v : graph.vertices()) {
      for (auto e : graph.in_edges(v)) {
        F& cur_message = state_.message(e);
        F new_message = state_.compute_message(e);
        residual += diff_(cur_message, new_message);
        update_(cur_message, std::move(new_message));
      }
    }
    return residual / graph.num_edges() / 2;
  }

  PairwiseBeliefState<F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
};

/**
 * Loopy BP schedule that updates the messages greedily based on the one with the largest current residual.
 */
template <typename F>
struct ResidualPropagationSchedule : PairwiseBeliefSchedule<F> {
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<F>::edge_descriptor;

  explicit ResidualPropagationSchedule(PairwiseBeliefState<F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
    initialize_residuals();
  }

  void initialize(const ShapeMap& shape_map) override {
    state_.initialize(shape_map);
    initialize_residuals();
  }

  real_type iterate() override {
    if (residuals_.empty()) return real_type(0);

    // extract the leading candidate edge
    edge_descriptor e = residuals_.pop().first;

    // update the message
    F& cur_message = state_.message(e);
    F new_message = state_.compute_message(e);
    real_type residual = diff_(cur_message, new_message);
    update_(cur_message, std::move(new_message));

    // recompute residual for dependent messages
    for (auto out : state_.graph().out_edges(e.target())) {
      if (out.target() != e.source()) {
        update_residual(out);
      }
    }
    update_residual(e);
    return residual;
  }

  /**
   * Computes the expected residual raised to n + 1.
   * \param alpha if 0, amounts to the average residual.
   */
  real_type expected_residual(real_type n = real_type(0)) const {
    real_type numer(0);
    real_type denom(0);
    const UndirectedGraph& graph = state_.graph();
    for (auto v : graph.vertices()) {
      for (auto e : graph.out_edges(v)) {
        real_type r = residual(e);
        numer += std::pow(r, n + 1);
        denom += std::pow(r, n);
      }
    }
    return numer / denom;
  }

  /**
   * Computes the maximum residual.
   */
  real_type maximum_residual() const {
    real_type result(0);
    const UndirectedGraph& graph = state_.graph();
    for (auto v : graph.vertices()) {
      for (auto e : graph.out_edges(v)) {
        result = std::max(result, residual(e));
      }
    }
    return result;
  }

private:
  void initialize_residuals() {
    const UndirectedGraph& graph = state_.graph();

    // Pass the flow along each directed edge
    for (auto v : graph.vertices()) {
      for (auto e : graph.in_edges(v)) {
        state_.message(e) = state_.compute_message(e);
      }
    }

    // Compute the residuals
    for (auto v : graph.vertices()) {
      for (auto e : graph.in_edges(v)) {
        update_residual(e);
      }
    }
  }

  void update_residual(edge_descriptor e) {
    real_type r = diff_(state_.message(e), state_.compute_message(e));
    if (!residuals_.contains(e)) {
      residuals_.push(e, r);
    } else {
      residuals_.update(e, r);
    }
  }

  real_type residual(edge_descriptor e) const {
    return residuals_.contains(e) ? residuals_.get(e) : real_type(0);
  }

  PairwiseBeliefState<F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
  MutableQueue<edge_descriptor, real_type> residuals_;
};

}
