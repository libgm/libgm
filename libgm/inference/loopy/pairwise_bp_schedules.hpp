#pragma once

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/inference/belief_defs.hpp>
#include <libgm/inference/loopy/pairwise_bp.hpp>

namespace libgm {

template <Argument Arg, typename F>
struct PairwiseBeliefSchedule {
  using real_type = typename F::real_type;
  virtual ~PairwiseBeliefSchedule() = default;
  virtual void initialize(std::function<F(Arg)> gen) = 0;
  virtual void initialize(const ShapeMap<Arg>& shape_map) = 0;
  virtual real_type iterate() = 0;
};

template <Argument Arg, typename F>
class SynchronousPropagationSchedule : public PairwiseBeliefSchedule<Arg, F> {
public:
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<Arg, F>::edge_descriptor;

  explicit SynchronousPropagationSchedule(PairwiseBeliefState<Arg, F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
    state_.swap(messages_);
    state_.initialize(gen);
  }

  void initialize(const ShapeMap<Arg>& shape_map) override {
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
  PairwiseBeliefState<Arg, F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
  ankerl::unordered_dense::map<edge_descriptor, F> messages_;
};

template <Argument Arg, typename F>
struct AsynchronousPropagationSchedule : PairwiseBeliefSchedule<Arg, F> {
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<Arg, F>::edge_descriptor;

  explicit AsynchronousPropagationSchedule(PairwiseBeliefState<Arg, F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
  }

  void initialize(const ShapeMap<Arg>& shape_map) override {
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

  PairwiseBeliefState<Arg, F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
};

template <Argument Arg, typename F>
struct ResidualPropagationSchedule : PairwiseBeliefSchedule<Arg, F> {
  using real_type = typename F::real_type;
  using edge_descriptor = typename PairwiseBeliefState<Arg, F>::edge_descriptor;

  explicit ResidualPropagationSchedule(PairwiseBeliefState<Arg, F>& state, BeliefUpdate<F> update, BeliefDiff<F> diff)
    : state_(state), update_(std::move(update)), diff_(std::move(diff)) {}

  void initialize(std::function<F(Arg)> gen) override {
    state_.initialize(gen);
    initialize_residuals();
  }

  void initialize(const ShapeMap<Arg>& shape_map) override {
    state_.initialize(shape_map);
    initialize_residuals();
  }

  real_type iterate() override {
    if (residuals_.empty()) return real_type(0);
    edge_descriptor e = residuals_.pop().first;
    F& cur_message = state_.message(e);
    F new_message = state_.compute_message(e);
    real_type residual = diff_(cur_message, new_message);
    update_(cur_message, std::move(new_message));

    for (auto out : state_.graph().out_edges(e.target())) {
      if (out.target() != e.source()) {
        update_residual(out);
      }
    }
    update_residual(e);
    return residual;
  }

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
    for (auto v : graph.vertices()) {
      for (auto e : graph.in_edges(v)) {
        state_.message(e) = state_.compute_message(e);
      }
    }
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

  PairwiseBeliefState<Arg, F>& state_;
  BeliefUpdate<F> update_;
  BeliefDiff<F> diff_;
  MutableQueue<edge_descriptor, real_type> residuals_;
};

} // namespace libgm
