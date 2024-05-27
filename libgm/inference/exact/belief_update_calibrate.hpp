#ifndef LIBGM_BELIEF_UPDATE_CALIBRATE_HPP
#define LIBGM_BELIEF_UPDATE_CALIBRATE_HPP

#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/traits/is_range.hpp>

namespace libgm {

/**
 * An algorithm for compute the marginal of a factorized probability model
 * using the division belief update algorithm on a junction tree.
 */
class BeliefUpdateCalibrate {

  // Public type declarations
  //--------------------------------------------------------------------------
public:
  struct Potential : Factor<
    One<Potential>,
    MultiplyJoinIn<Potential, Potential>,
    DIvideJoinIn<Potential, Potential>,
    Normalize<Potential>,
    Restrict<Potential, Vector>,
  > {};

  // Graph types
  using vertex_descriptor = ClusterGraph::vertex_descriptor;
  using edge_descriptor = ClusterGraph::edge_descriptor;

  Potential::VTable pt;
  Vector::VTable vt;
  ShapeMap shape_map;

  /// The underlying junction tree.
  ClusterGraphT<Potential, Potential> jt_;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. Constructs a belief update algorithm with no model.
   */
  BeliefUpdateCalibrate() { }

  /**
   * Constructs a belief update algorithm for the given collection of factors
   * whose product defines a probability distribution.
   * \tparam Range A forward range with elements convertible to F.
   */
  explicit BeliefUpdateCalibrate(Strategy strategy)
    : strategy(strategy) {}

  /**
   * Constructs a belief update algorithm to a junction tree whose ratio
   * of clique and separator potentials defines a probability distribution.
   */
  explicit BeliefUpdateCalibrate(const graph_type& jt)
    : jt_(jt) { }

  /**
   * Initializes the algorithm to the given network.
   */
  void reset(MarkovNetwork& mn) {
    // compute the junction tree for the given factors
    jt_.triangulated(mn, strategy);

    // intialize the clique and separator potentials to unity
    std::vector<size_t> vec;
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v] = Potential::one(v->shape(shape_map_, vec));
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e] = Potential::one(e->shape(shape_map_, vec));
    }
  }

  void multiply_in(const Domain& args, const Prototype& factor) {
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    assert(v);
    jt_[v].multiply_in(v->dims(domain, shape_map), factor, pt);
  }


  // Functions running the algorithm
  //--------------------------------------------------------------------------

  /**
   * Calibrates the junction tree by passing flow according to the message
   * passing protocol.
   */
  void calibrate() {
    mpp_traversal(jt_, nullptr, [&](const edge_descriptor& e) {
        jt_[e.target()].divide_in(jt_[e], target_dims(e), pt);
        jt_[e] = jt_[e.source()].marginal(jt_.source_dims(e, pt));
        jt_[e.target()].multiply_in(jt_[e], target_dims(e), pt);
      });
  }

  /**
   * Normalizes the clique and edge potentials.
   */
  void normalize() {
    for (vertex_descriptor v : jt_.vertices()) {
      jt_[v].normalize();
    }
    for (edge_descriptor e : jt_.edges()) {
      jt_[e].normalize();
    }
  }

  /**
   * Conditions the inference on an assignment to one or more variables
   * This is a mutable operation. Note that calibrate() needs to be called
   * afterwards.
   */
  void condition(const Assignment& a) {
    // Extract the restricted arguments
    Domain args = a.keys();

    // Update the factors and messages
    jt_.intersecting_clusters(args, [&](vertex_descriptor v) {
        Domain y, x; // restricted, retained
        v->cluster().partition(a, y, x);
        jt_[v] = jt_[v].restrict(v->dims(y, shape_map), Vector::values(a, y, vt), pt);
        jt_.update_cluster(v, x);
      });
    jt_.intersecting_separators(vars, [&](const edge_descriptor& e) {
        Domain y, x; // restricted, retained
        jt_.separator(e).partition(a, y, x);
        jt_[e] = jt_[e].restrict(v->dims(y, shape_map), Vector::values(a, y, vt), pt);
        jt_.update_separator(e, x);
      });
  }

  // Queries
  //--------------------------------------------------------------------------

  /// Returns the junction tree.
  const graph_type& jt() const {
    return jt_;
  }

  /// Returns the belief associated with a vertex.
  const Potential& belief(vertex_descriptor v) const {
    return jt_[v];
  }

  /// Returns the belief associated with an edge.
  const Potential& belief(const edge_descriptor& e) const {
    return jt_[e];
  }

  /**
   * Returns the belief for a set of arguments.
   * \throw std::invalid_argument
   *        if the specified set is not covered by a clique of
   *        the junction tree constructed by the engine.
   */
  Potential belief(const Domain& args) const {
    // Try to find a separator that covers the variables
    edge_descriptor e = jt_.find_separator_cover(args);
    if (e) {
      return jt_[e].marginal(e->dims(args, shape_map), pt);
    }

    // Next, look for a clique that covers the variables
    vertex_descriptor v = jt_.find_cluster_cover(args);
    if (v) {
      return jt_[v].marginal(v->dims(args, shape_map), pt);
    }

    // Did not find a suitable clique / separator
    throw std::invalid_argument(
      "belief: the domain is not covered by any clique or separator"
    );
  }

  // Private members
  //--------------------------------------------------------------------------
private:

  /**
   * Returns true if the potential arguments match the cliques and separators.
   */
  bool valid() const {
    for (vertex_descriptor v : jt.vertices()) {
      if (jt_[v].shape() != v->shape(shape_map_, vec)) { return false; }
    }
    for (edge_descriptor e : jt_.edges()) {
      if (jt_[e].shape() != e->shape(shape_map_, vec)) { return false; }
    }
    return true;
  }

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
public:
  BeliefUpdateCalibrateT() {
    /// Initialize the VTables.
  }

  void multiply_in(const Domain& args, const F& factor) {
    BeliefUpdateCalibrate::multiply_in(args, factor.cast_prototype<Potential>());
  }

  const Potential& belief(vertex_descriptor v) const {
    return BeliefUpdateCalibrate::belief(v).cast<F>();
  }

  const Potential& belief(edge_descriptor e) const {
    return BeliefUpdateCalibrate::belief(e).cast<F>();
  }

  F belief(const Domain& args) const {
    return BeliefUpdateCalibrate::belief(args).cast<F>();
  }
};


} // namespace libgm

#endif
