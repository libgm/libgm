#pragma once

#include <libgm/inference/exact/junction_tree_engine.hpp>
#include <libgm/model/cluster_graph.hpp>

namespace libgm {

/**
 * An algorithm for computing the marginals of a factorized probability model
 * using the multiplicative belief-update algorithm on a junction tree.
 *
 * \ingroup inference
 */
template <Argument Arg, typename F>
class BeliefUpdateCalibrate : public JunctionTreeEngine<Arg, F> {
public:
  using assignment_type = typename JunctionTreeEngine<Arg, F>::assignment_type;

  using vertex_descriptor = typename ClusterGraph<Arg>::vertex_descriptor;
  using edge_descriptor = typename ClusterGraph<Arg>::edge_descriptor;

  /// Initializes the algorithm to the cliques obtained by eliminating the given collection of factors.
  void reset(MarkovStructure<Arg> mg, const EliminationStrategy& strategy, const ShapeMap<Arg>& shape_map) override {
    jt_.triangulated(mg, strategy);
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v].reset(jt_.shape(v, shape_map));
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e].reset(jt_.shape(e, shape_map));
    }
  }

  /// Multiplies in a factor.
  void multiply_in(const Domain<Arg>& domain, const F& factor) override {
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    assert(v);
    jt_[v].multiply_in(factor, jt_.dims(v, domain));
  }

  /**
   * Conditions the inference on an assignment to one or more variables.
   * This is a mutable operation. Note that calibrate() needs to be called afterwards.
   */
  void condition(const assignment_type& a) override {
    Domain<Arg> args = a.keys();
    jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
      Domain<Arg> y, x;
      a.partition(jt_.cluster(v), y, x);
      jt_[v] = jt_[v].restrict_dims(jt_.dims(v, y), a.values(y));
      jt_.update_cluster(v, x);
    });
    jt_.intersecting_separators(args, [&](edge_descriptor e) {
      Domain<Arg> y, x;
      a.partition(jt_.separator(e), y, x);
      jt_[e] = jt_[e].restrict_dims(jt_.dims(e, y), a.values(y));
      jt_.update_separator(e, x);
    });
  }

  /// Performs inference by calibrating the junction tree.
  void calibrate() override {
    jt_.mpp_traversal(jt_.root(), [&](edge_descriptor e) {
      jt_[e.target()].divide_in(jt_[e], jt_.target_dims(e));
      jt_[e] = jt_[e.source()].marginal_dims(jt_.source_dims(e));
      jt_[e.target()].multiply_in(jt_[e], jt_.target_dims(e));
    });
  }

  /// Ensures that all the beliefs are normalized.
  void normalize() override {
    auto z = jt_[jt_.root()].marginal();
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v] /= z;
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e] /= z;
    }
  }

  /// Returns the underlying junction tree.
  const ClusterGraph<Arg, F, F>& jt() const { return jt_; }

  /// Returns the belief associated with a clique.
  const F& belief(vertex_descriptor v) const { return jt_[v]; }

  /// Returns the belief associated with a separator.
  const F& belief(edge_descriptor e) const { return jt_[e]; }

  /// Returns the belief for a set of variables.
  /// \throw std::invalid_argument if the specified set is not covered by any clique
  F belief(const Domain<Arg>& domain) const {
    edge_descriptor e = jt_.find_separator_cover(domain);
    if (e) {
      return jt_[e].marginal_dims(jt_.dims(e, domain));
    }
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    if (v) {
      return jt_[v].marginal_dims(jt_.dims(v, domain));
    }
    throw std::invalid_argument("BeliefUpdateCalibrate::belief: the domain is not covered by any clique or separator.");
  }

private:
  ClusterGraph<Arg, F, F> jt_;
}; // class BeliefUpdateCalibrate

} // namespace libgm
