#pragma once

#include <libgm/graph/cluster_graph.hpp>
#include <libgm/inference/exact/junction_tree_engine.hpp>

namespace libgm {

/**
 * An algorithm for compute the marginal of a factorized probability model
 * using the division belief update algorithm on a junction tree.
 *
 * \tparam F
 *         A type representing the factors. The type must support
 *         multiplication, division, and marginalization operations.
 */
template <typename F>
class BeliefUpdateCalibrate : public JunctionTreeEngine<F> {
public:
  // Graph types
  using vertex_descriptor = ClusterGraph::vertex_descriptor;
  using edge_descriptor = ClusterGraph::edge_descriptor;

  /// Initializes the algorithm to the given network.
  void reset(MarkovNetwork mn, const EliminationStrategy& strategy, const ShapeMap& shape_map) override {
    // compute the junction tree for the given factors
    jt_.triangulated(mn, strategy);

    // intialize the clique and separator potentials to unity
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v].reset(jt_.shape(v, shape_map));
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e].reset(jt_.shape(e, shape_map));
    }
  }

  /// Multiplies in the given factor to the underlying junction tree.
  void multiply_in(const Domain& domain, const F& factor) override {
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    assert(v);
    jt_[v].multiply_in(factor, jt_.dims(v, domain));
  }

  /// Conditions the inference on an assignment to one or more variables. This is a mutable operation.
  /// Note that calibrate() needs to be called afterwards.
  void condition(const typename F::assignment_type& a) override {
    // Extract the restricted arguments
    Domain args = a.keys();

    // Update the factors and messages
    jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
      Domain y, x; // restricted, retained
      a.partition(jt_.cluster(v), y, x);
      jt_[v] = jt_[v].restrict_dims(jt_.dims(v, y), a.values(y));
      jt_.update_cluster(v, x);
    });
    jt_.intersecting_separators(args, [&](edge_descriptor e) {
      Domain y, x; // restricted, retained
      a.partition(jt_.separator(e), y, x);
      jt_[e] = jt_[e].restrict_dims(jt_.dims(e, y), a.values(y));
      jt_.update_separator(e, x);
    });
  }

  /// Calibrates the junction tree by passing flow according to the message passing protocol.
  void calibrate() override {
    jt_.mpp_traversal(jt_.root(), [&](edge_descriptor e) {
      jt_[e.target()].divide_in(jt_[e], jt_.target_dims(e));
      jt_[e] = jt_[e.source()].marginal_dims(jt_.source_dims(e));
      jt_[e.target()].multiply_in(jt_[e], jt_.target_dims(e));
    });
  }

  /// Normalizes the clique and edge potentials.
  void normalize() override {
    auto z = jt_[jt_.root()].marginal();
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v] /= z;
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e] /= z;
    }
  }

  /// Returns the junction tree.
  const ClusterGraphT<F>& jt() const { return jt_; }

  /// Returns the node belief.
  const F& belief(vertex_descriptor v) const { return jt_[v]; }

  /// Returns the edge belief.
  const F& belief(edge_descriptor e) const { return jt_[e]; }

  /// Computes the belief for a set of arguments.
  /// \throw std::invalid_argument
  ///        if the specified set is not covered by a clique ofthe junction tree constructed by the engine.
  F belief(const Domain& domain) const {
    // Try to find a separator that covers the variables
    edge_descriptor e = jt_.find_separator_cover(domain);
    if (e) {
      return jt_[e].marginal_dims(jt_.dims(e, domain));;
    }

    // Next, look for a clique that covers the variables
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    if (v) {
      return jt_[v].marginal_dims(jt_.dims(v, domain));
    }

    // Did not find a suitable clique/separator
    throw std::invalid_argument(
      "BeliefUpdateCalibrate::belief: the domain is not covered by any clique or separator."
    );
  }

private:
  ClusterGraphT<F> jt_;
}; // class BeliefUpdateCalibrate

} // namespace libgm
