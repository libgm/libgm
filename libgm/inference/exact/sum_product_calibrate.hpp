#ifndef LIBGM_SUM_PRODUCT_CALIBRATE_HPP
#define LIBGM_SUM_PRODUCT_CALIBRATE_HPP

#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/graph/util/bidirectional.hpp>
#include <libgm/traits/is_range.hpp>

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
  struct Potential : Interfaces<
    Product<Factor, Factor>
  > {};

  // Graph types
  using graph_type  = cluster_graph<Factor, Bidirectional<Factor>>;
  using vertex_descriptor = typename graph_type::vertex_descriptor;
  using edge_descriptor   = typename graph_type::edge_descriptor;

  // Constructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. Constructs a sum-product algorithm with no model.
   */
  SumProductCalibrate() : calibrated_(true) { }

  /**
   * Initializes the algorithm to the iven junction tree that defines a
   * distribution via the product of the vertex properties.
   */
  void reset(const ClusterGraphT<Potential>& cg) {
    calibrated_ = false;
    assert(cg.tree());
    jt_.clear();

    // initialize the cliques and edges
    for (vertex_descriptor v : cg.vertices()) {
      jt_.add_vertex(v->cluster(), cg[v]);
    }
    for (edge_descriptor e : cg.edges()) {
      jt_.add_edge(jt_->vertex(e.source().index()), jt_->vertex(e.target().index()));
    }
  }

  /**
   * Initializes the algorithm to the given collection of factors.
   */
  void reset(const MarkovNetwork& mn) {
    calibrated_ = false;

    // initialize the junction tree
    jt_.triangulated(mn, strategy_);

    // Initialize the clique potentials
    for (id_t v : jt_.vertices()) {
      jt_[v] = Potential::one(jt_.shape(v, shape_map, vec), pt);
    }
  }

  /**
   * Multiplies in a factor.
   */
  void multiply_in(const Domain& domain, const Potential& factor) {
    vertex_descriptor v = jt_.find_cluster_cover(domain);
    assert(v);
    jt_[v].multiply_in(factor, v->dims(shape_map, domain), pt);
    calibrated_ = false;
  }

  // Function running the algorithm
  //--------------------------------------------------------------------------

  /**
   * Performs inference by calibrating the junction tree.
   */
  void calibrate() {
    mpp_traversal(jt_, nullptr, [&](const edge_descriptor& e) {
        Potential product = jt_[e.source()];
        for (edge_descriptor in : jt_.in_edges(e.source())) {
          if (in.source() != e.target()) {
            product.multiply_in(jt_[in](in), jt_.target_index(in), pt);
          }
        }
        jt_[e](e) = product.marginal(e->source_index());
      });
    calibrated_ = true;
  }

  /**
   * Ensures that all the beliefs are normalized.
   * The underlying junction tree must be calibrated.
   */
  void normalize() {
    assert(calibrated_ && !jt_.empty());
    // Compute the normalization constant z, and normalize the root
    // and every message in the direction from the root
    vertex_descriptor root = jt_.vertices().front();
    Result z = belief(root).sum(pt);
    jt_[root].divide_in(z, pt);
    pre_order_traversal(jt_, root, [&](const edge_descriptor& e) {
        jt_[e](e).divide_in(z, pt);
      });
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
        jt_.update_cluster(v, x); // FIXME: does this mess up the separators?
      });
    jt_.intersecting_separators(args, [&](const edge_descriptor& e) {
        Domain y, x; // restricted, retained
        e->separator().partition(a, y, x);
        Dims dims = e->dims(y, shape_map);
        Vector values = Vector::values(a, y, vt);
        jt_[e].forward = jt_[e].forward.restrict(dims, values, pt);
        jt_[e].reverse = jt_[e].reverse.restrict(dims, values, pt);
        jt_.update_separator(e, x);
      });

    // The junction tree needs to be calibrated afterwards
    calibrated_ = false;
  }

  // Queries
  //--------------------------------------------------------------------------

  //! Returns the underlying junction tree.
  const graph_type& jt() const {
    return jt_;
  }

  //! Returns the belief associated with a clique.
  Potential belief(vertex_descriptor v) const {
    assert(calibrated_);
    Potential result = jt_[v];
    for (edge_descriptor in : jt_.in_edges(v)) {
      result.multiply_in(in->target_dims(in), jt_[in](in), pt);
    }
    return result;
  }

  //! Returns the belief associated with a separator.
  Potential belief(const edge_descriptor& e) const {
    assert(calibrated_);
    return jt_[e].forward * jt_[e].reverse;
  }

  /**
   * Returns the belief for a set of variables.
   * \throw std::invalid_argument
   *        if the specified set is not covered by a clique of
   *        the junction tree constructed by the engine.
   */
  Potential belief(const Domain& args) const {
    assert(calibrated_);

    // Try to find a separator that covers the variables
    edge_descriptor e = jt_.find_separator_cover(args);
    if (e) {
      return belief(e).marginal(e->dims(args, shape_map), pt);
    }

    // Next, look for a clique that covers the variables
    vertex_descriptor v = jt_.find_cluster_cover(args);
    if (v) {
      return belief(v).marginal(v->dims(args, shape_map), pt);
    }

    // Did not find a suitable clique / separator
    throw std::invalid_argument(
      "belief: the domain is not covered by any clique or separator"
    );
  }

  //! Message along a directed edge
  const Potential& message(const edge_descriptor& e) const {
    return jt_[e](e);
  }

  //! Message along a directed edge.
  const Potential& message(vertex_descriptor u, vertex_descriptor v) const {
    return jt_(u, v)(u, v);
  }

  // Private data members
  //--------------------------------------------------------------------------
private:
  //! The junction tree used to store the cluster potentials and messages
  graph_type jt_;

  //! True if the inference has been performed
  bool calibrated_;

}; // class SumProductCalibrate

} // namespace libgm

#endif
