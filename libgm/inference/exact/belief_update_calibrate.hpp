#ifndef LIBGM_BELIEF_UPDATE_CALIBRATE_HPP
#define LIBGM_BELIEF_UPDATE_CALIBRATE_HPP

#include <libgm/graph/algorithm/make_clique.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/traits/is_range.hpp>

namespace libgm {

  /**
   * An algorithm for compute the marginal of a factorized probability model
   * using the division belief update algorithm on a junction tree.
   *
   * \tparam F the factor type
   * \ingroup inference
   */
  template <typename F>
  class belief_update_calibrate {

    // Public type declarations
    //==========================================================================
  public:
    // FactorizedInference types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::argument_type   argument_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           factor_type;
    typedef cluster_graph<domain_type, F, F> graph_type;

    // The descriptors for the junction tree
    typedef typename graph_type::vertex_type vertex_type;
    typedef typename graph_type::edge_type edge_type;

    // Constructors
    //==========================================================================
  public:
    /**
     * Default constructor. Constructs a belief update algorithm with no model.
     */
    belief_update_calibrate() { }

    /**
     * Constructs a belief update algorithm for the given collection of factors
     * whose product defines a probability distribution.
     * \tparam Range A forward range with elements convertible to F.
     */
    template <typename Range>
    explicit belief_update_calibrate(
        const Range& factors,
        typename std::enable_if<is_range<Range, F>::value>::type* = 0) {
      reset(factors);
    }

    /**
     * Constructs a belief update algorithm to a junction tree whose ratio
     * of clique and separator potentials defines a probability distribution.
     */
    explicit belief_update_calibrate(const graph_type& jt)
      : jt_(jt) { }

    /**
     * Initializes the algorithm to the given collection of factors.
     * \tparam Range A forward range with elements convertible to F.
     */
    template <typename Range>
    typename std::enable_if<is_range<Range, F>::value>::type
    reset(const Range& factors) {
      // compute the junction tree for the given factors
      undirected_graph<argument_type> mg;
      for (const F& factor : factors) {
        make_clique(mg, factor.arguments());
      }
      jt_.triangulated(mg, min_degree_strategy());

      // intialize the clique and separator potentials to unity
      for (vertex_type v : jt_.vertices()) {
        jt_[v] = F(jt_.cluster(v), result_type(1));
      }
      for (edge_type e : jt_.edges()) {
        jt_[e] = F(jt_.separator(e), result_type(1));
      }

      // multiply in the factors to cliques that cover them
      for (const F& factor : factors) {
        vertex_type v = jt_.find_cluster_cover(factor.arguments());
        assert(v);
        jt_[v] *= factor;
      }
    }

    /**
     * Initializes the algorithm to a junction tree whose ratio of clique
     * and separator potentials defines a probability distribution.
     */
    void reset(const graph_type& jt) {
      jt_ = jt;
      assert(valid());
    }

    // Functions running the algorithm
    //==========================================================================

    /**
     * Calibrates the junction tree by passing flow according to the message
     * passing protocol.
     */
    void calibrate() {
      mpp_traversal(jt_, id_t(), [&](const edge_type& e) {
          jt_[e.target()] /= jt_[e];
          jt_[e.source()].marginal(jt_.separator(e), jt_[e]);
          jt_[e.target()] *= jt_[e];
        });
    }

    /**
     * Normalizes the clique and edge potentials.
     */
    void normalize() {
      for (vertex_type v : jt_.vertices()) {
        jt_[v].normalize();
      }
      for (edge_type e : jt_.edges()) {
        jt_[e].normalize();
      }
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
          jt_[e] = jt_[e].restrict(a);
          jt_.update_separator(e, jt_[e].arguments());
        });
    }

    // Queries
    //==========================================================================

    //! Returns the junction tree.
    const graph_type& jt() const {
      return jt_;
    }

    /**
     * Returns the belief assocaited with a vertex.
     * The caller must not alter the belief arguments.
     */
    F& belief(vertex_type v) {
      return jt_[v];
    }

    //! Returns the belief associated with a vertex.
    const F& beliief(vertex_type v) const {
      return jt_[v];
    }

    //! Returns the belief associated with an edge.
    const F& belief(const edge_type& e) const {
      return jt_[e];
    }

    /**
     * Returns the belief for a set of variables.
     * \throw std::invalid_argument
     *        if the specified set is not covered by a clique of
     *        the junction tree constructed by the engine.
     */
    F belief(const domain_type& vars) const {
      // Try to find a separator that covers the variables
      edge_type e = jt_.find_separator_cover(vars);
      if (e) { return jt_[e].marginal(vars); }

      // Next, look for a clique that covers the variables
      vertex_type v = jt_.find_cluster_cover(vars);
      if (v) { return jt_[v].marginal(vars); }

      // Did not find a suitable clique / separator
      throw std::invalid_argument(
        "belief: the domain is not covered by any clique or separator"
      );
    }

    // Private members
    //==========================================================================
  private:

    /**
     * Returns true if the potential arguments match the cliques and separators.
     */
    bool valid() const {
      for (vertex_type v : jt_.vertices()) {
        if (jt_.cluster(v) != jt_[v].arguments()) { return false; }
      }
      for (edge_type e : jt_.edges()) {
        if (jt_.separator(e) != jt_[e].arguments()) { return false; }
      }
      return true;
    }

    //! The underlying junction tree.
    graph_type jt_;

  }; // class belief_update_calibrate

} // namespace libgm

#endif
