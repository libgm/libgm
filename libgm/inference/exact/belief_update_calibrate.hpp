#ifndef LIBGM_BELIEF_UPDATE_CALIBRATE_HPP
#define LIBGM_BELIEF_UPDATE_CALIBRATE_HPP

#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/traits/is_range.hpp>

namespace libgm {

  /**
   * An algorithm for compute the marginal of a factorized probability model
   * using the division belief update algorithm on a junction tree.
   *
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type representing the factors. The type must support
   *         multiplication, division, and marginalization operations.
   * \ingroup inference
   */
  template <typename Arg, typename F>
  class belief_update_calibrate {

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Graph types
    using graph_type  = cluster_graph<Arg, F, F>;
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
    //--------------------------------------------------------------------------
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
    explicit belief_update_calibrate(const Range& factors) {
      // TODO: some fancy ENABLE_IF
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
    void reset(const Range& factors) { // TODO: fancy enable_if
      // compute the junction tree for the given factors
      undirected_graph<Arg> mg;
      for (const auto& factor : factors) {
        mg.make_clique(factor.first);
      }
      jt_.triangulated(mg, min_degree_strategy());

      // intialize the clique and separator potentials to unity
      for (vertex_type v : jt_.vertices()) {
        jt_[v] = F(F::shape(jt_.cluster(v)), result_type(1));
      }
      for (edge_type e : jt_.edges()) {
        jt_[e] = F(F::shape(jt_.separator(e)), result_type(1));
      }

      // multiply in the factors to cliques that cover them
      for (const F& factor : factors) {
        vertex_type v = jt_.find_cluster_cover(factor.first);
        assert(v);
        jt_[v].dims(jt_.index(v, factor.first)) *= factor.second;
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
          jt_[e.target()].dims(target_index(e)) /= jt_[e];
          jt_[e] = jt_[e.source()].marginal(jt_.source_index(e));
          jt_[e.target()].dims(target_index(e)) *= jt_[e];
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
      jt_.intersecting_separators(vars, [&](const edge_type& e) {
          domain<Arg> y, x; // restricted, retained
          jt_.separator(e).partition(a, y, x);
          jt_[e] = jt_[e].restrict(jt_.index(v, y), a.values(y)).eval();
          jt_.update_separator(e, x);
        });
    }

    // Queries
    //==========================================================================

    //! Returns the junction tree.
    const graph_type& jt() const {
      return jt_;
    }

    /**
     * Returns the belief associated with a vertex.
     * The caller must not alter the shape.
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
     * Returns the belief for a set of arguments.
     * \throw std::invalid_argument
     *        if the specified set is not covered by a clique of
     *        the junction tree constructed by the engine.
     */
    F belief(const domain<Arg>& args) const {
      // Try to find a separator that covers the variables
      edge_type e = jt_.find_separator_cover(args);
      if (e) {
        return jt_[e].marginal(jt_.index(e, args));
      }

      // Next, look for a clique that covers the variables
      vertex_type v = jt_.find_cluster_cover(args);
      if (v) {
        return jt_[v].marginal(jt_.index(v, args));
      }

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
        if (jt_[v].shape() != F::shape(jt_.cluster(v))) { return false; }
      }
      for (edge_type e : jt_.edges()) {
        if (jt_.[e].shape() != F::shape(jt_.separator(e))) { return false; }
      }
      return true;
    }

    //! The underlying junction tree.
    graph_type jt_;

  }; // class belief_update_calibrate

} // namespace libgm

#endif
