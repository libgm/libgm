#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/graph/markov_network.hpp>
#include <libgm/graph/undirected_edge.hpp>

#include <ankerl/unordered_dense.h>

namespace libgm {

/**
 * An interface for storing the belief state and computing the messages and beliefs in pairwise belief propagation.
 */
template <typename NodeF>
struct PairwiseBeliefState {
  virtual ~PairwiseBeliefState() {}

  /// Initializes all the messages using the given generator (must be provided).
  virtual void initialize(std::function<NodeF(Arg)> gen) = 0;

  /// Initializes all the messages by initializing them to identity using the provided shape_map.
  virtual void initialize(const ShapeMap& map) = 0;

  /// Computes the message along an edge.
  virtual NodeF compute_message(UndirectedEdge<Arg> e) const = 0;

  /// Returns a message. Throws std::out_of_range if not already present.
  virtual const NodeF& message(UndirectedEdge<Arg> e) const = 0;

  /// Returns a mutable reference ot the message.
  virtual NodeF& message(UndirectedEdge<Arg> e) = 0;

  /// Returns the underlying Markov network.
  virtual const MarkovNetwork& graph() = 0;

  /// Swaps the underlying message map for another one.
  virtual void swap(ankerl::unordered_dense::map<UndirectedEdge<Arg>, NodeF>& other) = 0;
};

/**
 * An engine that performs loopy belief propagation. If the underlying markov network changes,
 * the results are undefined. The lifetime of the Markov network object must extend past the
 * lifetime of this object.
 *
 * \ingroup inference
 */
template <typename NodeF, typename EdgeF>
class PairwiseBeliefPropagation : public PairwiseBeliefState<NodeF> {
public:
  using real_type = typename NodeF::real_type;

  /// Constructs a loopy bp engine for the given graph.
  PairwiseBeliefPropagation(const MarkovNetworkT<NodeF, EdgeF>& graph)
    : graph_(graph) {}

  void initialize(std::function<NodeF(Arg)> gen) override {
    for (Arg v : graph_.vertices()) {
      for (auto e : graph_.in_edges(v)) {
        message(e) = gen(v);
      }
    }
  }

  void initialize(const ShapeMap& shape_map) override {
    for (Arg v : graph_.vertices()) {
      for (auto e : graph_.in_edges(v)) {
        message(e) = NodeF(shape_map(v));
      }
    }
  }

  NodeF compute_message(UndirectedEdge<Arg> e) const override {
    Arg u = e.source();
    Arg v = e.target();

    // Compute the product of factor at u and messages into u.
    NodeF incoming = factor(u);
    for (auto f : graph_.in_edges(u)) {
      if (f.source() != v) {
        incoming *= message(f);
      }
    }

    // Sum-product over the edge
    NodeF result = e.is_nominal()
      ? graph_[e].multiply_front(incoming).marginal_back(1)
      : graph_[e].multiply_back(incoming).marginal_front(1);

    // Normalize and return
    result.normalize();
    return result;
  }

  const NodeF& message(UndirectedEdge<Arg> e) const override {
    return messages_.at(e);
  }

  NodeF& message(UndirectedEdge<Arg> e) override {
    return messages_[e];
  }

  const MarkovNetwork& graph() override {
    return graph_;
  }

  void swap(ankerl::unordered_dense::map<UndirectedEdge<Arg>, NodeF>& other) override {
    using std::swap;
    swap(messages_, other);;
  }

  /// Computes the node belief.
  NodeF belief(Arg u) const {
    NodeF f = factor(u);
    for (auto e : graph_.in_edges(u)) {
      f *= message(e);
    }
    f.normalize();
    return f;
  }

  /// Computes the edge belief.
  EdgeF belief(UndirectedEdge<Arg> e) const {
    Arg u = e.source();
    Arg v = e.target();
    NodeF fu = factor(u);
    NodeF fv = factor(v);
    for (auto e : graph_.in_edges(u)) {
      if (e.source() != v) { fu *= message(e); }
    }
    for (auto e : graph_.in_edges(v)) {
      if (e.source() != u) { fv *= message(e); }
    }
    EdgeF result = factor(e);
    result.multiply_in_front(fu);
    result.multiply_in_back(fv);
    result.normalize();
    return result;
  }

private:
  const NodeF& factor(Arg u) const {
    return graph_[u];
  }

  EdgeF factor(UndirectedEdge<Arg> e) const {
    return e.is_nominal() ? graph_[e] : graph_[e].transpose();
  }

  /// A reference to the Markov network used in the computations.
  const MarkovNetworkT<NodeF, EdgeF>& graph_;

  /// The underlying state.
  ankerl::unordered_dense::map<UndirectedEdge<Arg>, NodeF> messages_;
};

}
