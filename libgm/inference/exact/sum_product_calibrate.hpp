#pragma once

#include <libgm/factor/concepts.hpp>
#include <libgm/graph/util/bidirectional.hpp>
#include <libgm/inference/exact/junction_tree_engine.hpp>
#include <libgm/model/cluster_graph.hpp>

namespace libgm {

/**
 * An algorithm for computing the marginals of a factorized probability model
 * using the multiplicative sum-product algorithm on a junction tree.
 *
 * \ingroup inference
 */
template <Argument Arg, typename F>
class SumProductCalibrate : public JunctionTreeEngine<Arg, F> {
public:
  using assignment_type = typename JunctionTreeEngine<Arg, F>::assignment_type;

  // Descriptors
  using vertex_descriptor = typename ClusterGraph<Arg>::vertex_descriptor;
  using edge_descriptor = typename ClusterGraph<Arg>::edge_descriptor;

  /// Default constructor. Constructs a sum-product algorithm with no model.
  SumProductCalibrate() = default;

  /// Initializes the algorithm to the given junction tree that defines a distribution via the product of the vertex
  /// properties.
  template <typename Other>
  void reset(const ClusterGraph<Arg, F, Other>& cg) {
    calibrated_ = false;
    jt_.clear();

    ankerl::unordered_dense::map<vertex_descriptor, vertex_descriptor> map;
    for (vertex_descriptor v : cg.vertices()) {
      map.emplace(v, jt_.add_vertex(cg.cluster(v), cg[v]));
    }
    for (edge_descriptor e : cg.edges()) {
      jt_.add_edge(map.at(e.source()), map.at(e.target()));
    }

    assert(jt_.is_tree());
  }

  /// Initializes the algorithm to the cliques obtained by eliminating given collection of factors.
  void reset(MarkovStructure<Arg> mg, const EliminationStrategy& strategy, const ShapeMap<Arg>& shape_map) override {
    calibrated_ = false;

    jt_.triangulated(mg, strategy);

    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v].reset(jt_.shape(v, shape_map));
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e].forward.reset(jt_.shape(e, shape_map));
      jt_[e].reverse.reset(jt_.shape(e, shape_map));
    }
  }

  /// Multiplies in a factor.
  void multiply_in(const Domain<Arg>& domain, const F& factor) override {
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    assert(v);
    jt_[v].multiply_in(factor, jt_.dims(v, domain));
    calibrated_ = false;
  }

  // Function running the algorithm
  //--------------------------------------------------------------------------

  /// Performs inference by calibrating the junction tree.
  void calibrate() override {
    jt_.mpp_traversal(jt_.root(), [&](edge_descriptor e) {
      F factor = jt_[e.source()];
      for (edge_descriptor in : jt_.in_edges(e.source())) {
        if (in.source() != e.target()) {
          factor.multiply_in(message(in), jt_.target_dims(in));
        }
      }
      message(e) = factor.marginal_dims(jt_.source_dims(e));
    });
    calibrated_ = true;
  }

  /// Ensures that all the beliefs are normalized. The underlying junction tree must be calibrated.
  void normalize() override {
    assert(calibrated_ && !jt_.empty());

    vertex_descriptor root = jt_.root();
    auto z = belief(root).marginal();
    jt_[root] /= z;
    jt_.pre_order_traversal(root, [this, z](edge_descriptor e) {
      message(e) /= z;
    });
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
      Dims dims = jt_.dims(e, y);
      typename F::value_list values = a.values(y);
      jt_[e].forward = jt_[e].forward.restrict_dims(dims, values);
      jt_[e].reverse = jt_[e].reverse.restrict_dims(dims, values);
      jt_.update_separator(e, x);
    });

    calibrated_ = false;
  }

  // Queries
  //--------------------------------------------------------------------------

  /// Returns the underlying junction tree.
  const ClusterGraph<Arg, F, Bidirectional<F>>& jt() const {
    return jt_;
  }

  /// Returns the belief associated with a clique.
  F belief(vertex_descriptor v) const {
    assert(calibrated_);
    F result = jt_[v];
    for (edge_descriptor in : jt_.in_edges(v)) {
      result.multiply_in(message(in), jt_.target_dims(in));
    }
    return result;
  }

  /// Returns the belief associated with a separator.
  F belief(edge_descriptor e) const {
    assert(calibrated_);
    return jt_[e].forward * jt_[e].reverse;
  }

  /// Returns the belief for a set of variables.
  /// \throw std::invalid_argument if the specified set is not covered by any clique
  F belief(const Domain<Arg>& domain) const {
    assert(calibrated_);

    edge_descriptor e = jt_.find_separator_cover(domain);
    if (e) {
      return belief(e).marginal_dims(jt_.dims(e, domain));
    }

    vertex_descriptor v = jt_.find_cluster_cover(domain);
    if (v) {
      return belief(v).marginal_dims(jt_.dims(v, domain));
    }

    throw std::invalid_argument("SumProductCalibrate::belief: the domain is not covered by any clique or separator");
  }

  /// Potentials and messages
  const F& potential(vertex_descriptor v) {
    return jt_[v];
  }

  F& message(edge_descriptor e) {
    return jt_[e](e);
  }

  const F& message(edge_descriptor e) const {
    return jt_[e](e);
  }

private:
  /// The junction tree used to store the factors and messages
  ClusterGraph<Arg, F, Bidirectional<F>> jt_;

  /// True if the inference has been performed
  bool calibrated_ = false;
}; // class SumProductCalibrate

} // namespace libgm
