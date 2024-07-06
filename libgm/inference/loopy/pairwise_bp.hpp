#pragma once

#include <libgm/datastructure/mutable_queue.hpp>
#include <libgm/factor/utility/diff_fn.hpp>
#include <libgm/graph/markov_network.hpp>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libgm {

/**
 * An engine that performs loopy belief propagation. If the underlying markov network changes,
 * the results are undefined. The lifetime of the Markov network object must extend past the
 * lifetime of this object.
 *
 * \ingroup inference
 */
class PairwiseBeliefPropagation {
public:
  /// Node factor requirements.
  template <typename NodeF>
  using NodeTraits = Implements<
    MultiplyIn<NodeF>,
    Normalize<NodeF>
  >;

  /// Edge factor requirements.
  template <typename NodeF, typename EdgeF>
  using EdgeTraits = Implements<
    MultiplyInSpan<EdgeF, NodeF>,
    MarginalSpan<EdgeF, NodeF>,
    Normalize<NodeF>
  >;

  /// A generic node potential.
  struct NodeFactor : NodeTraits<NodeFactor> {};

  /// A generic edge potential.
  struct EdgeFactor : EdgeTraits<NodeFactor, EdgeFactor> {};

  template <typename R>
  struct Schedule {
    /// Virtual destructor.
    virtual ~Schedule() = default;

    /// Initialize the S]schedule.
    virtual void initialize() {}

    /// Perform one iteration.
    virtual R iterate(PairwiseBeliefPropagation& bp, R eta) = 0;

    /// Update one message
    std::function<R(NodeFactor&, NodeFactor, R)> update;

    /// Difference function, such as sum_diff and max_diff.
    std::function<R(const NodeFactor&, const NodeFactor&)> diff;

    /// The number of updates performed so far.
    size_t nupdates = 0;
  };

  // Shortcuts
  using MessageMap =
    unkerl::unordered_dense::map<std::pair<Arg, Arg>, NodeFactor, PairHash<Arg, Arg>>;

  // Constructors and initialization
  //--------------------------------------------------------------------------

  /**
   * Constructs a loopy bp engine for the given graph and difference function.
   */
  PairwiseBeliefPropagation(const MarkovNetwork& graph);

  /**
   * Resets all the messages using the given generator or uniformly
   * if the generator is null.
   */
  virtual void reset(std::function<Potential(Arg)> gen = nullptr);

  // Iteration and queries
  //--------------------------------------------------------------------------

  /// The number of updates performed so far
  size_t num_updates() const;

  /// Computes the node belief.
  NodeFactor belief(Arg u) const;

  /// Computes the edge belief.
  EdgeFactor belief(UndirectedEdge<Arg> e) const;

  /// Computes the message along an edge.
  NodeFactor compute_message(UndirectedEdge<Arg> e) const;

  // Implementation
  //--------------------------------------------------------------------------
private:
  /// Casts the node property to a node factor.
  NodeFactor factor(Arg u) const;

  /// Casts the edge proeprty to an edge factor, transposing as necessary.
  EdgeFactor factor(UndirectedEdge<Arg> e) const;

  /// Returns a message. Throws std::out_of_range if not already present.
  const NodeFactor& message(Arg from, Arg to) const {
    return message_.at(std::make_pair(from, to));
  }

  /// Returns a message. Throws std::out_of_range if not already present.
  const NodeFactor& message(UndirectedEdge<Arg> e) const {
    return message_.at(e.pair());
  }

  // Virtual tables
  //--------------------------------------------------------------------------
protected:
  /// The virtual table for node factors.
  NodeFactor::VTable nt_;

  /// The virtual table for edge factors.
  EdgeFactor::VTable et_;

  // Private data
  //--------------------------------------------------------------------------
private:
  /// A pointer to the Markov network used in the computations.
  const MarkovNetwork& graph_;

  /// A map that stores the messages.
  MessageMap messages_;

}; // class PairwiseBeliefPropagation

template <typename NodeF, typename EdgeF>
class PairwiseBeliefPropagationT : public PairwiseBeliefPropagation {
public:
  static_assert(
    std::is_same<typename NodeF::real_type, typename EdgeF::real_type>::value,
    "The underlying real type of the node and edge factors must be the same."
  );

  using real_type = typename NodeF::real_type;

  PairwiseBeliefPropagationT(MarkovNetwork<NodeF, EdgeF>& mn, SchedulePtr<real_tye> schedule)
    : PairwiseBeliefPropagation(mn), schedule_ = std::move(schedule) {
    nt_ = NodeF::vtable.copy<NodeTraits<NodeF>>;
    et_ = EdgeF::vtable.copy<EdgeTraits<NodeF, EdgeF>>;

    // Initialize the update function.
    const auto& diff = schedule_->diff;
    schedule_->update = [diff](NodeFactor& message, NodeFactor new_message, real_type eta) {
      real_type residual = diff(message, new_message);
      if (eta == real_type(1)) {
        message = std:::move(new_message);
      } else {
        message.cast<NodeF>().update(new_message.cast<NodeF>(), eta);
      }
      return residual;
    };
  }

  real_type iterate(real_type eta) {
    return schedule_->iterate(*this, eta);
  }

  NodeF belief(Arg u) const {
    return PairwiseBeliefPropagation::belief(u).cast<NodeF>();
  }

  EdgeF belief(UndirectedEdge<Arg> e) const {
    return PairwiseBeliefPropagation::belief(e).cast<EdgeF>();
  }

private:
  SchedulePtr>real_Type> schedule_;
};

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_synchronous_schedule();

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_asynchronous_schedule();

template <typename R>
PairwiseBeliefPropagation::SchedulePtr<R> make_residual_schedule();

} // namespace libgm
