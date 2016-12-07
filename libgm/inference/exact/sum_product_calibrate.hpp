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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type representing the factors. The type must support
   *         multiplication and marginalization operations.
   * \ingroup inference
   */
  template <typename Arg, typename F>
  class sum_product_calibrate {

    // Public types
    //--------------------------------------------------------------------------
  public:
    // Graph types
    using graph_type  = cluster_graph<Arg, F, bidirectional<F> >;
    using vertex_type = typename graph_type::vertex_type;
    using edge_type   = typename graph_type::edge_type;

    // Argument types
    using argument_type     = Arg;
    using argument_hasher   = typename argument_traits<Arg>::hasher;
    using argument_iterator = typename graph_type::argument_iterator;

    // Factor types
    using real_type   = typename F::real_type;
    using result_type = typename F::result_type;
    using factor_type = F;

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
    explicit sum_product_calibrate(const Range& factors) {
      reset(factors);
    }

    /**
     * Constructs a sum-product algorithm for a junction tree whose vertices
     * are associated with factors s.t. the product of the factors represents
     * a probability distribution.
     */
    explicit sum_product_calibrate(const cluster_graph<domain_type, F>& jt) {
      reset_graph(jt);
    }

    /**
     * Initializes the algorithm to the given collection of factors.
     * \tparm Range A forward range with elements convertible to F
     */
    template <typename Range>
    void reset(const Range& factors) {
      calibrated_ = false;

      // initialize the junction tree
      undirected_graph<Arg> mg;
      for (const auto& factor : factors) {
        mg.make_clique(factor.first);
      }
      jt_.triangulated(mg, min_degree_strategy());

      // Initialize the clique potentials
      for (id_t v : jt_.vertices()) {
        jt_[v] = F(F::shape(jt_.cluster(v)), result_type(1));
      }
      for (const auto& factor : factors) {
        vertex_type v = jt_.find_cluster_cover(factor.first);
        assert(v);
        jt_[v].dims(jt_.index(v, factor.first)) *= factor.second;
      }
    }

    /**
     * Initializes the algorithm to the iven junction tree that defines a
     * distribution via the product of the vertex properties.
     */
    void reset_graph(const cluster_graph<Arg, F>& jt) {
      calibrated_ = false;
      assert(jt.tree());
      jt_.clear();

      // initialize the cliques and edges
      for (id_t v : jt.vertices()) {
        assert(F::shape(jt.cluster(v)) == jt[v].shape());
        jt_.add_cluster(v, jt.cluster(v), jt[v]);
      }
      for (edge_type e : jt.edges()) {
        jt_.add_edge(e.source(), e.target());
      }
    }

    // Function running the algorithm
    //==========================================================================

    /**
     * Performs inference by calibrating the junction tree.
     */
    void calibrate() {
      mpp_traversal(jt_, id_t(), [&](const edge_type& e) {
          F product = jt_[e.source()];
          for (edge_type in : jt_.in_edges(e.source())) {
            if (in.source() != e.target()) {
              product.dims(jt_.target_index(in)) *= jt_[in](in);
            }
          }
          jt_[e](e) = product.marginal(jt_.source_index(e));
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
      vertex_type root = jt_.vertices().front();
      result_type z = belief(root).sum();
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
    void condition(const assignment<Arg, real_type>& a) {
      // Extract the restricted arguments
      domain<Arg> args = a.keys();

      // Update the factors and messages
      jt_.intersecting_clusters(args, [&](vertex_type v) {
          domain<Arg> y, x; // restricted, retained
          jt_.cluster(v).partition(a, y, x);
          jt_[v] = jt_[v].restrict(jt_.index(v, y), a.values(y)).eval();
          jt_.update_cluster(v, x);
        });
      jt_.intersecting_separators(args, [&](const edge_type& e) {
          domain<Arg> y, x; // restricted, retained
          jt_.separator(e).partition(a, y, x);
          uint_vector index = jt_.index(v, y);
          auto values = a.values(y);
          jt_[e].forward = jt_[e].forward.restrict(index, values).eval();
          jt_[e].reverse = jt_[e].reverse.restrict(index, values).eval();
          jt_.update_separator(e, x);
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
        result.dims(jt_.target_index(in)) *= jt_[in](in);
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
    F belief(const domain<Arg>& args) const {
      assert(calibrated_);

      // Try to find a separator that covers the variables
      edge_type e = jt_.find_separator_cover(args);
      if (e) {
        return belief(e).marginal(jt_.index(e, args));
      }

      // Next, look for a clique that covers the variables
      vertex_type v = jt_.find_cluster_cover(args);
      if (v) {
        return belief(v).marginal(jt_.index(v, args));
      }

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
      return jt_(u, v)(u, v);
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
