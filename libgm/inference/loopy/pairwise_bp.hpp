#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/shape.hpp>
#include <libgm/model/markov_network.hpp>

#include <ankerl/unordered_dense.h>

namespace libgm {

/**
 * An interface for storing the belief state and computing the messages and beliefs in pairwise belief propagation.
 */
template <Argument Arg, typename NodeF>
struct PairwiseBeliefState {
  using edge_descriptor = typename MarkovNetwork<Arg>::edge_descriptor;

  virtual ~PairwiseBeliefState() = default;

  /// Initializes all the messages using the given generator (must be provided).
  virtual void initialize(std::function<NodeF(Arg)> gen) = 0;

  /// Initializes all the messages by initializing them to identity using the provided shape_map.
  virtual void initialize(const ShapeMap<Arg>& map) = 0;

  /// Computes the message along an edge.
  virtual NodeF compute_message(edge_descriptor e) const = 0;

  /// Returns a message. Throws std::out_of_range if not already present.
  virtual const NodeF& message(edge_descriptor e) const = 0;

  /// Returns a mutable reference ot the message.
  virtual NodeF& message(edge_descriptor e) = 0;

  /// Returns the underlying graph view.
  virtual const UndirectedGraph& graph() const = 0;

  /// Swaps the underlying message map for another one.
  virtual void swap(ankerl::unordered_dense::map<edge_descriptor, NodeF>& other) = 0;
};

/**
 * An engine that performs loopy belief propagation. If the underlying markov network changes,
 * the results are undefined. The lifetime of the Markov network object must extend past the
 * lifetime of this object.
 *
 * \ingroup inference
 */
template <Argument Arg, typename NodeF, typename EdgeF>
class PairwiseBeliefPropagation : public PairwiseBeliefState<Arg, NodeF> {
public:
  using real_type = typename NodeF::real_type;
  using graph_type = MarkovNetwork<Arg, NodeF, EdgeF>;
  using vertex_descriptor = typename graph_type::vertex_descriptor;
  using edge_descriptor = typename PairwiseBeliefState<Arg, NodeF>::edge_descriptor;

  /// Constructs a loopy bp engine for the given graph.
  explicit PairwiseBeliefPropagation(const graph_type& graph)
    : graph_(graph) {}

  void initialize(std::function<NodeF(Arg)> gen) override {
    for (vertex_descriptor v : graph_.vertices()) {
      for (auto e : graph_.in_edges(v)) {
        message(e) = gen(graph_.argument(v));
      }
    }
  }

  void initialize(const ShapeMap<Arg>& shape_map) override {
    for (vertex_descriptor v : graph_.vertices()) {
      for (auto e : graph_.in_edges(v)) {
        message(e) = NodeF(shape_map(graph_.argument(v)));
      }
    }
  }

  NodeF compute_message(edge_descriptor e) const override {
    vertex_descriptor u = e.source();
    vertex_descriptor v = e.target();

    // Compute the product of factor at u and messages into u.
    NodeF incoming = factor(u);
    for (auto f : graph_.in_edges(u)) {
      if (f.source() != v) {
        incoming *= message(f);
      }
    }

    // Sum-product over the edge
    NodeF result = graph_.is_nominal(e)
      ? graph_[e].multiply_front(incoming).marginal_back(1)
      : graph_[e].multiply_back(incoming).marginal_front(1);

    // Normalize and return
    result.normalize();
    return result;
  }

  const NodeF& message(edge_descriptor e) const override {
    return messages_.at(e);
  }

  NodeF& message(edge_descriptor e) override {
    return messages_[e];
  }

  const UndirectedGraph& graph() const override {
    return graph_.graph();
  }

  void swap(ankerl::unordered_dense::map<edge_descriptor, NodeF>& other) override {
    using std::swap;
    swap(messages_, other);
  }

  /// Computes the node belief.
  NodeF belief(Arg u) const {
    return belief(graph_.vertex(u));
  }

  NodeF belief(vertex_descriptor u) const {
    NodeF f = factor(u);
    for (auto e : graph_.in_edges(u)) {
      f *= message(e);
    }
    f.normalize();
    return f;
  }

  /// Computes the edge belief.
  EdgeF belief(edge_descriptor e) const {
    vertex_descriptor u = e.source();
    vertex_descriptor v = e.target();
    NodeF fu = factor(u);
    NodeF fv = factor(v);
    for (auto in : graph_.in_edges(u)) {
      if (in.source() != v) { fu *= message(in); }
    }
    for (auto in : graph_.in_edges(v)) {
      if (in.source() != u) { fv *= message(in); }
    }
    EdgeF result = factor(e);
    result.multiply_in_front(fu);
    result.multiply_in_back(fv);
    result.normalize();
    return result;
  }

private:
  const NodeF& factor(vertex_descriptor u) const {
    return graph_[u];
  }

  EdgeF factor(edge_descriptor e) const {
    return graph_.is_nominal(e) ? graph_[e] : graph_[e].transpose();
  }

  /// A reference to the Markov network used in the computations.
  const graph_type& graph_;

  /// The underlying state.
  ankerl::unordered_dense::map<edge_descriptor, NodeF> messages_;
};

} // namespace libgm
