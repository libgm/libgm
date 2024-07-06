#include "pairwise_bp.hpp"

namespace libgm {

PairwiseBeliefPropagation::PairwiseBeliefPropagation(MarkovNetwork& mn, real_binary_fn<NodeF> diff)
  : graph_(graph), diff_(std::move(diff)), nupdates_(0) {
  reset();
}

void PairwiseBeliefPropagation::reset(std::function<Potential(Arg)> gen = nullptr) {
  for (vertex_descriptor v : graph_.vertices()) {
    for (edge_descriptor e : graph_.in_edges(v)) {
      if (gen) {
        message(e) = gen(v);
      } else {
        message(e) = NodeFactor::ones(shape_map(v));
      }
    }
  }
}

size_t PairwiseBeliefPropagation::num_updates() const {
  return nupdates_;
}

NodeFactor PairwiseBeliefPropagation::belief(Arg u) const {
  NodeFactor f = factor(u);
  for (Arg v : graph_.adjacent_vertices(u)) {
    f *= message(v, u);
  }
  f.normalize();
  return f;
}

EdgeFactor PairwiseBeliefPropagation::belief(UndirectedEdge<Arg> e) const {
  Arg u = e.source();
  Arg v = e.target();
  NodeFactor fu = factor(u);
  NodeFactor fv = factor(v);
  for (Arg w : graph_.adjacent_vertices(u)) {
    if (w != v) { fu *= message(w, u); }
  }
  for (Arg w : graph_.adjacent_vertices(v)) {
    if (w != u) { fv *= message(w, v); }
  }
  EdgeFactor result = factor(e);
  result.multiply_in_front(fu);
  result.multiply_in_back(fv);
  result.normalize();
  return result;
}

NodeFactor PairwiseBeliefPropagation::factor(vertex_descriptor v) const {
  return graph_[v].cast<NodeFactor>(nt_);
}

EdgeFactor PairwiseBeliefPropagation::factor(edge_descriptor e) const {
  return e.forward() ? graph[e].cast<EdgeFactor>(et_) : graph[e].cast<EdgeFactor>(et_).transpose();
}

const NodeFactor& PairwiseBeliefPropagation::message(Arg from, Arg to) const {
  return message_.at(std::make_pair(from, to));
}

const NodeFactor& PairwiseBeliefPropagation::message(UndirectedEdge<Arg> e) const {
  return message_.at(e.pair());
}

NodeFactor PairwiseBeliefPropagation::compute_message(UndirectedEdge<Arg> e) const {
  Arg u = e.source();
  Arg v = e.target();

  // Compute the product of factor at u and messages into u.
  NodeFactor incoming = factor(u);
  for (Arg w : graph_.adjacent_vertices(u)) {
    if (w != v) {
      incoming *= message(w, u);
    }
  }

  // Sum-product over the edge
  EdgeFactor edge_f = graph_[e].cast<EdgeFactor>(et_);
  NodeFactor result = e.forward()
    ? edge_f.multiply_in_front(incoming).marginal_back(1)
    : edge_f.multiply_in_back(incoming).marginal_front(1);

  // Normalize and return
  result.normalize();
  return result;
}

// Schedules
//--------------------------------------------------------------------------

template <typename R>
using Schedule = PairwiseBeliefPropagation::Schedule<R>;

/**
 * Loopy BP schedule that updates the messages synchronously.
 *
 * \tparam R the underlying real type of the factor
 */
template <typename R>
struct SynchronousPropagationSchedule : Schedule<R> {
  R iterate(PairwiseBeliefPropagation& bp, R eta) override {
    R residual(0);
    for (UndirectedEdge<Arg> e : bp.graph().edges()) {
      UndirectedEdge<Arg> f = e.reverse();
      residual = std::max(residual, this->update(message(e), bp.compute_message(e), eta));
      residual = std::max(residual, this->update(message(f), bp.compute_message(f), eta));
      this->nupdates += 2;
    }
    bp.swap(new_message_);
    return residual;
  }

  PairwiseBeliefPropagation::Node& message(UndirectedEdge<Arg> e) override {
    return new_messages[e.pair()];
  }

  MessageMap new_messages;
};

/**
 * Loopy BP schedule that updates the messages in a round-robin manner.
 *
 * \tparam R the underlying real type of the factor.
 */
template <typename R>
struct AsynchronousPropagationSchedule : Schedule<R> {
  R iterate(PairwiseBeliefPropagation& bp, R eta) override {
    R residual(0);
    for (UndirectedEdge<Arg> e : bp.graph().edges()) {
      UndirectedEdge<Arg> f = e.reverse();
      residual = std::max(residual, this->update(bp.message(e), bp.compute_message(e), eta));
      residual = std::max(residual, this->update(bp.message(f), bp.compute_message(f), eta));
      this->nupdates += 2;
    }
    return residual;
  }
};

/**
 * Loopy BP schedule that updates the messages greedily based on the one
 * with the largest current residual.
 *
 * \tparam R the underlying real type of the factor
 */
template <typename R>
struct ResidualPropagationSchedule : Schedule<R> {
  void initialize(PairwiseBeliefPropagation& bp) override {
    // Pass the flow along each directed edge
    for (UndirectedEdge<Arg> e : bp.graph().edges()) {
      bp.message(e) = bp.compute_message(e);
      bp.message(e.reverse()) = bp.compute_message(e.reverse());
    }

    // Compute the residuals
    for (UndirectedEdge<Arg> e : bp.graph().edges()) {
      update_residual(e);
      update_residual(e.reverse());
    }
  }

  R iterate(PairwiseBeliefPropagation& bp, R eta) override {
    if (q_.empty()) return R(0);

    // extract the leading candidate edge
    UndirectedEdge<Arg> e = q_.pop().first;

    // update message(e) and recompute residual for dependent messages
    R residual = this->update(bp.message(e), bp.compute_message(e), eta);
    this->nupdates++;
    for (auto out : bp.graph().out_edges(e.target())) {
      if (out.target() != e.source()) {
        update_residual(out);
      }
    }
    if (eta != R(1)) {
      update_residual(e);
    }
    return residual;
  }

  R residual(UndirectedEdge<Arg> e) const {
    try {
      return q_.get(e));
    } catch (std::out_of_range&) {
      return R(0);
    }
  }

  /// Updates the residuals for an edge.
  void update_residual(UndirectedEdge<Arg> e) {
    R r = this->diff(this->message(e), this->compute_message(e));
    if (!q_.contains(e)) {
      q_.push(e, r);
    } else {
      q_.update(e, r);
    }
  }

  /**
   * Computes the expected residual raised to n + 1.
   * \param alpha if 0, amounts to the average residual.
   */
  R expected_residual(real_type n = real_type(0)) const {
    real_type numer(0);
    real_type denom(0);
    for (edge_descriptor e : graph_.edges()) {
      real_type r = residual(e);
      numer += std::pow(r, n + 1);
      denom += std::pow(r, n);
      real_type s = residual(e.reverse());
      numer += std::pow(s, n + 1);
      denom += std::pow(s, n);
    }
    return numer / denom;
  }

  /**
   * Computes the maximum residual.
   */
  R maximum_residual() const {
    real_type result(0);
    for (edge_descriptor e : graph_.edges()) {
      result = std::max(result, residual(e));
      result = std::max(result, residual(e.reverse()));
    }
    return result;
  }

  // A queue of residuals.
  MutableQueue<std::pair<Arg, Arg>, R> q_;
};

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_synchronous_schedule() {
  return std::make_unique<SynchronousPropagationSchedule>();
}

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_asynchronous_schedule() {
  return std::make_unique<AsynchronousPropagationSchedule>();
}

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_residual_schedule() {
  return std::make_unique<ResidualPropagationSchedule>();
}

} // namespace libgm