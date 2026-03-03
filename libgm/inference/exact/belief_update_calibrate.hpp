#pragma once

#include <libgm/graph/cluster_graph.hpp>

namespace libgm {

/**
 * An algorithm for compute the marginal of a factorized probability model
 * using the division belief update algorithm on a junction tree.
 */
class BeliefUpdateCalibrate {

  // Public type declarations
  //--------------------------------------------------------------------------
public:
  struct PotentialCref : Implements<...> { void* ref; };
  struct PotentialRef : PotentialCref, Implements<...> { PotentialRef& operator=(...); };
  struct Potential : PotentialRef { std::unique_ptr<char[]> data; };

  // Graph types
  using vertex_descriptor = ClusterGraph::vertex_descriptor;
  using edge_descriptor = ClusterGraph::edge_descriptor;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. Constructs a belief update algorithm with no model.
   */
  BeliefUpdateCalibrate() = default;

  /**
   * Initializes the algorithm to the given network.
   */
  void reset(MarkovNetwork& mn, const EliminationStrategy& strategy, const ShapeMap& shape_map);

  /**
   * Multiplies in the given factor to the underlying junction tree.
   */
  void multiply_in(const Domain& domain, Potential factor);

  // Functions running the algorithm
  //--------------------------------------------------------------------------

  /**
   * Calibrates the junction tree by passing flow according to the message
   * passing protocol.
   */
  void calibrate();

  /**
   * Normalizes the clique and edge potentials.
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

  /// Returns the junction tree.
  const ClusterGraph& jt() const {
    return jt_;
  }

  /// Returns the belief associated with a vertex.
  PotentialCref belief(vertex_descriptor v) const {
    return {jt_[v], vtable};
  }

  /// Returns the belief associated with an edge.
  PotentialCref belief(edge_descriptor e) const {
    return {jt_[e], vtable};
  }

  /**
   * Returns the belief for a set of arguments.
   * \throw std::invalid_argument
   *        if the specified set is not covered by a clique of
   *        the junction tree constructed by the engine.
   */
  Potential belief(const Domain& domain) const;

protected:
  std::unique_ptr<Potential> one(const Shape& shape) const = 0;

  Potential::VTable vtable;
}; // class BeliefUpdateCalibrate

)
/**
 * \tparam F
 *         A type representing the factors. The type must support
 *         multiplication, division, and marginalization operations.
 * \ingroup inference
 */
template <typename F>
class BeliefUpdateCalibrateT : public BeliefUpdateCalibrate {
protected:
  /// The underlying junction tree.
  ClusterGraphT<F, F> jt_;

  Potential one(const Shape& shape) const override {
    return F(shape);
  }

public:
  BeliefUpdateCalibrateT() {
    vtable = F::vtable.generic(); // exposiiton only
  }

  void multiply_in(const Domain& args, const F& factor) {
    BeliefUpdateCalibrate::multiply_in(args, PotentialCref(&factor));
  }

  const F& belief(vertex_descriptor v) const {
    return BeliefUpdateCalibrate::belief(v).cast<F>();
  }

  const F& belief(edge_descriptor e) const {
    return BeliefUpdateCalibrate::belief(e).cast<F>();
  }

  F belief(const Domain& args) const {
    return BeliefUpdateCalibrate::belief(args).cast<F>();
  }
};

} // namespace libgm
