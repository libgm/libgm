#ifndef LIBGM_SUM_PRODUCT_CALIBRATE_HPP
#define LIBGM_SUM_PRODUCT_CALIBRATE_HPP

#include <libgm/graph/algorithm/make_clique.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/bidirectional.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/graph/undirected_graph.hpp>
#include <libgm/traits/is_range.hpp>

namespace libgm {

  /** 
   * An algorithm for computing the marginals of a factorized probability model
   * using the multiplicative sum-product algorithm on a junction tree.
   *
   * \tparam F the factor type
   * \ingroup inference
   */
  template <typename F>
  class sum_product_calibrate {

    // Public type declarations    
    //==========================================================================
  public:
    // FactorizedInference types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           factor_type;
    typedef cluster_graph<domain_type, F, bidirectional<F> > graph_type;

    // The descriptors for the junction tree 
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;

    // Constructors
    //==========================================================================
  public:
    /**
     * Default constructor. Constructs a sum-product algorithm with no model.
     */
    sum_product_calibrate() : calibrated_(true) { }

    /**
     * Constructs a sum-product algorithm for the given collection of factors
     * whose product represents a probability distribution.
     * \tparam Range A forward range with elements convertible to F
     */
    template <typename Range>
    explicit sum_product_calibrate(
        const Range& factors,
        typename std::enable_if<is_range<Range, F>::value>::type* = 0) {
      reset(factors);
    }

    /**
     * Constructs a sum-product algorithm for a junction tree whose vertices
     * are associated with factors s.t. the product of the factors represents
     * a probability distribution.
     */
    explicit sum_product_calibrate(const cluster_graph<domain_type, F>& jt) {
      reset(jt);
    }

    /**
     * Initializes the algorithm to the given collection of factors.
     * \tparm Range A forward range with elements convertible to F
     */
    template <typename Range>
    typename std::enable_if<is_range<Range, F>::value>::type
    reset(const Range& factors) {
      calibrated_ = false;

      // initialize the junction tree
      undirected_graph<variable_type> mg;
      for (const F& factor : factors) {
        make_clique(mg, factor.arguments());
      }
      jt_.triangulated(mg, min_degree_strategy());

      // Initialize the clique potentials
      for (vertex_type v : jt_.vertices()) {
        jt_[v] = F(jt_.cluster(v), result_type(1));
      }
      for (const F& factor : factors) {
        size_t v = jt_.find_cluster_cover(factor.arguments());
        assert(v);
        jt_[v] *= factor;
      }
    }

    /** 
     * Initializes the algorithm to the iven junction tree that defines a
     * distribution via the product of the vertex properties.
     */
    void reset(const cluster_graph<domain_type, F>& jt) {
      calibrated_ = false;
      assert(jt_.tree());
      jt_.clear();

      // initialize the cliques and edges
      for (vertex_type v : jt_.vertices()) {
        assert(jt_.cluster(v) == jt[v].arguments());
        jt_.add_cluster(v, jt_.cluster(v), jt[v]);
      }
      for (edge_type e : jt_.edges()) {
        jt_.add_edge(e.source(), e.target());
      }
    }

    // Function running the algorithm
    //==========================================================================

    /**
     * Performs inference by calibrating the junction tree.
     */
    void calibrate() { 
      mpp_traversal(jt_, 0, [&](const edge_type& e) {
          F product = jt_[e.source()];
          for (edge_type in : jt_.in_edges(e.source())) {
            if (in.source() != e.target()) {
              product *= jt_[in](in);
            }
          }
          jt_[e](e) = product.marginal(jt_.separator(e));
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
      vertex_type root = *jt_.vertices().begin();
      result_type z = belief(root).marginal();
      jt_[root] /= z;
      pre_order_traversal(jt_, root, [&](const edge_type& e) {
          jt_[e](e) /= z;
        });
    }

    /**
     * Conditions the inference on an assignment to one or more variables
     * This is a mutable operation. Note that calibrate() needs to be called
     * afterwards.
     */
    void condition(const assignment_type& a) {
      // Extract the restricted arguments
      domain_type vars;
      for (const auto& p : a) { vars.push_back(p.first); }

      // Update the factors and messages
      jt_.intersecting_clusters(vars, [&](vertex_type v) {
          jt_[v] = jt_[v].restrict(a);
          jt_.update_cluster(v, jt_[v].arguments());
        });
      jt_.intersecting_separators(vars, [&](const edge_type& e) {
          jt_[e].forward = jt_[e].forward.restrict(a);
          jt_[e].reverse = jt_[e].reverse.restrict(a);
          jt_.update_separator(e, jt_[e].forward.arguments());
        });

      // The junction tree needs to be calibrated afterwards
      calibrated_ = false;
    }

    // Queries
    //==========================================================================

    //! Returns the underlying junction tree.
    const graph_type& jt() const {
      return jt_;
    }

    //! Returns the belief associated with a clique.
    F belief(vertex_type v) const {
      assert(calibrated_);
      F result = jt_[v];
      for (edge_type in : jt_.in_edges(v)) {
        result *= jt_[in](in);
      }
      return result;
    }

    //! Returns the belief associated with a separator.
    F belief(const edge_type& e) const {
      assert(calibrated_);
      return jt_[e].forward * jt_[e].reverse;
    }

    /**
     * Returns the belief for a set of variables.
     * \throw std::invalid_argument 
     *        if the specified set is not covered by a clique of 
     *        the junction tree constructed by the engine.
     */
    F belief(const domain_type& vars) const {
      assert(calibrated_);

      // Try to find a separator that covers the variables
      edge_type e = jt_.find_separator_cover(vars);
      if (e) { return belief(e).marginal(vars); }

      // Next, look for a clique that covers the variables
      vertex_type v = jt_.find_cluster_cover(vars);
      if (v) { return belief(v).marginal(vars); }

      // Did not find a suitable clique / separator
      throw std::invalid_argument(
        "belief: the domain is not covered by any clique or separator"
      );
    }

    //! Message along a directed edge
    const F& message(const edge_type& e) const {
      return jt_[e](e);
    }

    //! Message along a directed edge.
    const F& message(vertex_type u, vertex_type v) const {
      return jt_(u,v)(u,v);
    }

    // Private data members
    //==========================================================================
  private:
    //! The junction tree used to store the cluster potentials and messages
    graph_type jt_;

    //! True if the inference has been performed
    bool calibrated_;

  }; // class sum_product_calibrate

} // namespace libgm

#endif
