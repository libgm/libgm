#pragma once

#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/graph/util/bidirectional.hpp>

namespace libgm {

/**
 * An algorithm for computing the marginals of a factorized probability model
 * using the multiplicative sum-product algorithm on a junction tree.
 *
 * \ingroup inference
 */
class SumProductCalibrate {
  // Public types
  //--------------------------------------------------------------------------
public:
  /// Factor traits.
  template <typename F, typename R>
  using FactorTraits = Implements<
    Multiply<F, F>,
    MultiplyInDims<F, F>,
    MarginalDims<F>
  > {};

  /// A generic factor.
  struct Factor : FactorTraits<Factor> {};

  // Descriptors
  using vertex_descriptor = typename ClusterGraph::vertex_descriptor;
  using edge_descriptor   = typename ClusterGraph::edge_descriptor;

  // A function that divides its argument by stored constant.
  using Normalizer = std::function<void(Factor&)>;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. Constructs a sum-product algorithm with no model.
   */
  SumProductCalibrate();

  /**
   * Initializes the algorithm to the given junction tree that defines a
   * distribution via the product of the vertex properties.
   */
  void reset(const ClusterGraph& cg);

  /**
   * Initializes the algorithm to the cliques obtained by eliminating given collection of factors.
   */
  void reset(MarkovNetworkT<>& mn, const EliminationStreategy& strategy, const ShapeMap& shape_map);

  /**
   * Multiplies in a factor.
   */
  void multiply_in(const Domain& domain, const Factor& factor);

  // Function running the algorithm
  //--------------------------------------------------------------------------

  /**
   * Performs inference by calibrating the junction tree.
   */
  void calibrate();

  /**
   * Ensures that all the beliefs are normalized.
   * The underlying junction tree must be calibrated.
   */
  void normalize();

  /**
   * Conditions the inference on an assignment to one or more variables
   * This is a mutable operation. Note that calibrate() needs to be called
   * afterwards.
   */
  void condition(const Assignment& a);

  // Queries
  //--------------------------------------------------------------------------

  /// Returns the underlying junction tree.
  const ClusterGraph& jt() const {
    return jt_;
  }

  /// Returns the belief associated with a clique.
  Factor belief(vertex_descriptor v) const;

  /// Returns the belief associated with a separator.
  Factor belief(edge_descriptor e) const;

  /**
   * Returns the belief for a set of variables.
   * \throw std::invalid_argument if the specified set is not covered by any clique
   */
  Factor belief(const Domain& domain) const;

  /// Message along a directed edge
  const Factor& message(edge_descriptor e) const {
    return jt_[e](e);
  }

protected:
  // Returns a normalizer capturing the normalization constant of belief at given vertex.
  virtual Normalizer normalizer(vertex_descriptor v) const = 0;

  // Private data members
  //--------------------------------------------------------------------------
private:
  /// The junction tree used to store the factors and messages
  ClusterGraphT<Factor, Bidirectional<Factor>> jt_;

  /// True if the inference has been performed
  bool calibrated_;

  /// The VTable for the factors.
  Factor::VTable vt_;

}; // class SumProductCalibrate

template <typename F>
class SumProductCalibrateT : public SumProductCalibrate {
public:
  SumProductCalibrateT() {
    vtable = F::vtable.copy<FactorTraits<F, typename F::result_type>>();
  }

  void reset(const ClusterGraphT<F>& cg) {
    SumProductCalibrate::reset(cg);
  }

  void multiply_in(const Domain& domain, const F& factor) {
    SumProductCalibrate::multiply_in(factor.cast<Factor>(vtable));
  }

  F belief(vertex_descriptor v) const {
    return SumProductCalibrate::belief(v).cast<F>();
  }

  F belief(edge_descriptor e) const {
    return SumProductCalibrate::belief(e).cast<F>();
  }

  F belief(const Domain& domain) const {
    return SumProductCalibrate::belief(domain).cast<F>();
  }

  const F& message(edge_descriptor e) const {
    return SumProductCalibrate::message(e).cast<F>();
  }

  Normalizer normalizer(vertex_descriptor v) const override {
    typename F::result_type z = belief(v).sum();
    return [z](Factor& factor) { factor.cast<F>() /= z; };
  }
};

} // namespace libgm
