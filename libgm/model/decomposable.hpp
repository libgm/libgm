#ifndef LIBGM_DECOMPOSABLE_HPP
#define LIBGM_DECOMPOSABLE_HPP

#include <libgm/factor/util/operations.hpp>
#include <libgm/graph/algorithm/min_degree_strategy.hpp>
#include <libgm/graph/algorithm/tree_traversal.hpp>
#include <libgm/graph/cluster_graph.hpp>
#include <libgm/inference/exact/variable_elimination.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/range/iterator_range.hpp>

#include <functional>
#include <iterator>
#include <sstream>
#include <vector>

namespace libgm {

  /**
   * A decomposable representation of a probability distribution.
   * Conceptually, decomposable model is a junction tree, in which
   * each vertex and each edge is associated with a factor.
   * The distribution is equal to to the product of clique marginals,
   * divided by the product of separator marginals.
   *
   * \tparam F A type representing the factors. The type must support
   *           multiplication and division operations.
   *
   * \ingroup model
   */
  template <typename F>
  class decomposable {

    typedef cluster_graph<typename F::domain_type, F, F> graph_type;

    // Public type declarations
    //==========================================================================
  public:
    // FactorizedModel types
    typedef typename F::real_type       real_type;
    typedef typename F::result_type     result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           value_type;
    class /* forward declaration */     const_iterator;
    typedef const_iterator              iterator;

    // Graph vertex, edge, and properties
    typedef id_t                  vertex_type;
    typedef undirected_edge<id_t> edge_type;
    typedef F                       vertex_property;
    typedef F                       edge_property;

    // Graph iterators
    typedef typename graph_type::vertex_iterator   vertex_iterator;
    typedef typename graph_type::neighbor_iterator neighbor_iterator;
    typedef typename graph_type::edge_iterator     edge_iterator;
    typedef typename graph_type::in_edge_iterator  in_edge_iterator;
    typedef typename graph_type::out_edge_iterator out_edge_iterator;

    // Model iterators
    typedef typename graph_type::argument_iterator argument_iterator;

    // Constructors and destructors
    //==========================================================================
  public:
    /**
     * Default constructor. The distribution has no arguments and
     * is identically one.
     */
    decomposable() { }

    //! Swaps two decomposable models in place.
    friend void swap(decomposable& a, decomposable& b) {
      swap(a.jt_, b.jt_);
    }

    //! Serialize members.
    void save(oarchive& ar) const {
      ar << jt_;
    }

    //! Deserialize members
    void load(iarchive& ar) {
      ar >> jt_;
    }

    //! Returns true if the two decomposable models are identical.
    friend bool operator==(const decomposable& a, const decomposable& b) {
      return a.jt_ == b.jt_;
    }

    //! Returns true if the two decomposable models are not identical.
    friend bool operator!=(const decomposable& a, const decomposable& b) {
      return a.jt_ != b.jt_;
    }

    //! Prints the decomposable model to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const decomposable& dm) {
      out << dm.jt_;
      return out;
    }

    // Accessors
    //==========================================================================
    //! Returns the null vertex, guaranteed to be id_t().
    static id_t null_vertex() {
      return id_t();
    }

    //! Returns the range of all vertices (clique ids) of the model.
    iterator_range<vertex_iterator>
    vertices() const {
      return jt_.vertices();
    }

    //! Returns the first vertex or the null vertex if the graph is empty.
    id_t root() const {
      return jt_.empty() ? id_t() : *jt_.vertices().begin();
    }

    //! Returns the vertices (clique ids) adjacent to u.
    iterator_range<neighbor_iterator>
    neighbors(id_t u) const {
      return jt_.neighbors(u);
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return jt_.edges();
    }

    //! Returns the edges incoming to a vertex.
    iterator_range<in_edge_iterator>
    in_edges(id_t u) const {
      return jt_.in_edges(u);
    }

    //! Returns the outgoing edges from a vertex.
    iterator_range<out_edge_iterator>
    out_edges(id_t u) const {
      return jt_.out_edges(u);
    }

    //! Returns true if the graph contains the given vertex.
    bool contains(id_t u) const {
      return jt_.contains(u);
    }

    //! Returns true if the graph contains an undirected edge {u, v}.
    bool contains(id_t u, id_t v) const {
      return jt_.contains(u, v);
    }

    //! Returns true if the graph contains an undirected edge.
    bool contains(const edge_type& e) const {
      return jt_.contains(e);
    }

    //! Returns an undirected edge (u, v). The edge must exist.
    edge_type edge(id_t u, id_t v) const {
      return jt_.edge(u, v);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t in_degree(id_t u) const {
      return jt_.in_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t out_degree(id_t u) const {
      return jt_.out_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex.
    std::size_t degree(id_t u) const {
      return jt_.degree(u);
    }

    //! Returns true if the graph has no vertices / no arguments.
    bool empty() const {
      return jt_.empty();
    }

    //! Returns the number of vertices.
    std::size_t num_vertices() const {
      return jt_.num_vertices();
    }

    //! Returns the number of edges.
    std::size_t num_edges() const {
      return jt_.num_edges();
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
    edge_type reverse(const edge_type& e) const {
      return e.reverse();
    }

    //! Returns the range of arguments of the model.
    iterator_range<argument_iterator> arguments() const {
      return jt_.arguments();
    }

    //! Returns the number of arguments in the model.
    std::size_t num_arguments() const {
      return jt_.num_arguments();
    }

    //! Returns the clique associated with a vertex.
    const domain_type& clique(id_t v) const {
      return jt_.cluster(v);
    }

    //! Returns the separator associated with an edge.
    const domain_type& separator(const edge_type& e) const {
      return jt_.separator(e);
    }

    //! Returns the iterator to the first factor.
    const_iterator begin() const {
      return const_iterator(this);
    }

    //! Returns the iterator to the one past the last factor.
    const_iterator end() const {
      return const_iterator();
    }

    //! Returns the marginal associated with a vertex.
    const F& operator[](id_t u) const {
      return jt_[u];
    }

    //! Returns the marginal associated with an edge.
    const F& operator[](const edge_type& e) const {
      return jt_[e];
    }

    //! Returns the underlying junction tree.
    const graph_type& jt() const {
      return jt_;
    }

    // Queries
    //==========================================================================

    /**
     * Computes the Markov graph capturing the dependencies in this model.
     */
    void markov_graph(undirected_graph<variable_type>& mg) const {
      for (variable_type v : arguments()) {
        mg.add_vertex(v);
      }
      for (id_t v : vertices()) {
        make_clique(mg, clique(v));
      }
    }

    /**
     * Returns tre if this decomposable model is valid.
     * Decomposable model is vlaid if the underlying junction tree satisfies
     * the running intersection property, and the cliques and separators
     * match the corresponding marginal.
     *
     * \param msg if not null, an object where the error message is stored
     */
    bool valid(std::string* msg = nullptr) const {
      if (!jt_.tree()) {
        if (msg) { *msg = "The underlying graph is not a tree"; }
        return false;
      }
      if (!jt_.running_intersection()) {
        if (msg) { *msg = "The underlying graph does not satisfiy RIP"; }
        return false;
      }
      for (id_t v : vertices()) {
        if (clique(v) != jt_[v].arguments()) {
          if (msg) {
            std::ostringstream out;
            out << "Inconsistent clique and factor arguments: "
                << clique(v) << " != " << jt_[v].arguments();
            *msg = out.str();
          }
          return false;
        }
      }
      for (edge_type e : edges()) {
        if (separator(e) != jt_[e].arguments()) {
          if (msg) {
            std::ostringstream out;
            out << "Incosistent separator and factor arguments: "
                << separator(e) << " != " << jt_[e].arguments();
            *msg = out.str();
          }
          return false;
        }
      }
      return true;
    }

    /**
     * Computes a marginal over an arbitrary subset of variables.
     * The variables must be all present in this decomposble model.
     */
    F marginal(const domain_type& domain) const {
      if (domain.empty()) {
        return F(typename F::result_type(1));
      }

      // Look for a separator that covers the variables.
      edge_type e = jt_.find_separator_cover(domain);
      if (e) { return jt_[e].marginal(domain); }

      // Look for a clique that covers the variables.
      id_t v = jt_.find_cluster_cover(domain);
      if (v) { return jt_[v].marginal(domain); }

      // Otherwise, compute the factors whose product represents
      // the marginal
      std::list<F> factors;
      marginal(domain, factors);
      return prod_all(factors).marginal(domain); // TODO: should be reoder()
    }

    /**
     * Computes a list of factors whose product represents
     * a marginal over a subset of variables.
     */
    void marginal(const domain_type& domain, std::list<F>& factors) const {
      factors.clear();
      if (domain.empty()) return;

      const_cast<graph_type&>(jt_).mark_subtree_cover(domain, false);
      for (id_t v : vertices()) {
        if (jt_.marked(v)) {
          factors.push_back(jt_[v]);
        }
      }
      for (edge_type e : edges()) {
        if (jt_.marked(e)) {
          factors.push_back(typename F::result_type(1) / jt_[e]);
        }
      }

      variable_elimination(factors, domain, sum_product<F>());
    }

    /**
     * Computes a decomposable model that represents the marginal
     * distribution over one ore more variables.
     * Note: This operation can create large cliques.
     */
    void marginal(const domain_type& domain, decomposable& result) const {
      std::list<F> factors;
      marginal(domain, factors);
      result.reset(factors);
    }

    /**
     * Computes the entropy of the distribution represented by this
     * decomposable model.
     */
    real_type entropy() const {
      real_type result(0);
      for (id_t v : vertices()) { result += jt_[v].entropy(); }
      for (edge_type e : edges()) { result -= jt_[e].entropy(); }
      return result;
    }

    /**
     * Computes the entropy over a subset of variables.
     */
    real_type entropy(const domain_type& domain) const {
      // first try to compute the entropy directly from the marginals
      edge_type e = jt_.find_separator_cover(domain);
      if (e) { return jt_[e].entropy(domain); }

      id_t v = jt_.find_cluster_cover(domain);
      if (v) { return jt_[v].entropy(domain); }

      // failing that, compute the marginal of the model
      decomposable tmp;
      marginal(domain, tmp);
      return tmp.entropy();
    }

    /**
     * Computes the conditional entropy H(Y | X), where Y, X are subsets
     * of the arguments of this model.
     * \todo see if we can optimize this
     */
    real_type conditional_entropy(const domain_type& y,
                                  const domain_type& x) const {
      return entropy(x | y) - entropy(x);
    }

    /**
     * Computes the mutual information I(A ; B) between two subsets of*
     * arguments of this model.
     */
    real_type mutual_information(const domain_type& a,
                                 const domain_type& b) const {
      return entropy(a) + entropy(b) - entropy(a | b);
    }

    /**
     * Computes the conditional mutual information I(A; B | C),
     * where A,B,C must be subsets of the arguments of this model.
     * This is computed using I(A; B | C) = H(A | C) - H(A | B, C).
     *
     * @param base   Base of logarithm.
     * @return double representing the conditional mutual information.
     */
    real_type conditional_mutual_information(const domain_type& a,
                                             const domain_type& b,
                                             const domain_type& c) const {
      return conditional_entropy(a, c) - conditional_entropy(a, b | c);
    }

    /**
     * Compute the maximum probability and stores the corresponding
     * assignment to a.
     */
    result_type maximum(assignment_type& a) const {
      a.clear();
      if (empty()) { return result_type(1); }

      // copy the clique marginals into factors
      std::unordered_map<id_t, F> factor;
      for (id_t v : vertices()) {
        factor[v] = jt_[v];
      }

      // collect evidence
      post_order_traversal(jt_, root(), [&](const edge_type& e) {
          factor[e.target()] *= factor[e.source()].maximum(separator(e));
          factor[e.target()] /= jt_[e];
        });

      // extract the maximum for the root clique
      result_type result = jt_[root()].maximum(a);

      // distribute evidence
      pre_order_traversal(jt_, root(), [&](const edge_type& e) {
          factor[e.target()].restrict(a).maximum(a);
        });

      return result;
    }

    /**
     * Draws a random sample from this model.
     * \tparam Generator a random number generator.
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      a.clear();
      jt_[root()].sample(rng, a);
      pre_order_traversal(jt_, root(), [&](const edge_type& e) {
          jt_[e.target()].restrict(a).sample(rng, a);
        });
    }

    /**
     * Returns the log-likelihood of the given assignment.
     * If the assignment includes all the arguments of this model,
     * this function computes the complete log-likelihood.
     * Otherwise, this function computes the marginal log-likelihood.
     */
    real_type log(const assignment_type& a) const {
      domain_type args = restricted_args(a);
      real_type result(0);
      if (args.size() == num_arguments()) {
        for (id_t v : vertices()) { result += jt_[v].log(a); }
        for (edge_type e : edges()) { result -= jt_[e].log(a); }
      } else {
        std::list<F> factors;
        marginal(args, factors);
        for (const F& factor : factors) { result += factor.log(a); }
      }
      return result;
    }

    /**
     * Returns the probability of the assignment.
     * if the assignment includes all the arguments of this model,
     * this function computes the joint probaiblity p(a).
     * Otherwise, this function computes the marginal probability.
     */
    result_type operator()(const assignment_type& a) const {
      return std::exp(log(a));
    }

    // Restructuring operations
    //==========================================================================

    //! Clears all factors and variables from this model.
    void clear() {
      jt_.clear();
    }

    /**
     * Initializes the decomposable model to the product of the given
     * factors.
     * \tparam Range a single pass range with elements convertible to F
     */
    template <typename Range>
    void reset(const Range& factors) {
      clear();
      operator*=(factors);
    }

    /**
     * Initializes the decomposable model to the given range of marginals.
     * The argument domains of the marginals must be triangulated.
     * \tparam Range a single pass range with elements convertible to F
     */
    template <typename Range>
    void reset_marginals(const Range& marginals) {
      clear();

      // initialize the clique marginals and the tree structure
      for (const F& marginal : marginals) {
        jt_.add_cluster(marginal.arguments(), marginal);
      }
      jt_.mst_edges();

      // compute the separator marginals
      for (edge_type e : edges()) {
        if (clique(e.source()).size() < clique(e.target()).size()) {
          jt_[e] = jt_[e.source()].marginal(separator(e));
        } else {
          jt_[e] = jt_[e.target()].marginal(separator(e));
        }
      }

      // calibrate & normalize in case the marginals are not consistent
      calibrate();
      normalize();
    }

    /**
     * Initializes the decomposable model to a single marginal.
     */
    void reset_marginal(const F& marginal) {
      clear();
      jt_.add_cluster(marginal.arguments(), marginal);
    }

    /**
     * Restructures this decomposable model so that it includes the
     * supplied cliques. These cliques can include new variables
     * (which are not present in this model). In this case, the
     * marginals over these new variables will be set to 1.
     *
     * \tparam Range A single pass range with elements convertible to Domain
     *
     * \todo right now we retriangulate the entire model, but only
     *       the subtree containing the parameter vars must be
     *       retriangulated.
     */
    template <typename Range>
    void retriangulate(const Range& cliques) {
      // Create a graph with the cliques of this decomposable model.
      undirected_graph<variable_type> mg;
      markov_graph(mg);

      // Add the new cliques
      for (const domain_type& clique : cliques) {
        make_clique(mg, clique);
      }

      // Now create a new junction tree for the Markov graph and
      // initialize the clique/separator marginals
      graph_type jt;
      jt.triangulated(mg, min_degree_strategy());
      for (id_t v : jt.vertices()) {
        jt[v] = F(jt.cluster(v), typename F::result_type(1));
        jt[v] *= marginal(intersecting_args(jt.cluster(v)));
      }
      for (edge_type e : jt.edges()) {
        jt[e] = F(jt.separator(e), typename F::result_type(1));
        jt[e] = marginal(intersecting_args(jt.separator(e)));
      }

      // Swap in the new junctino tree
      swap(jt, jt_);
    }

    /**
     * Restructures this decomposable model so that it has a clique
     * that covers the supplied variables, and returns the vertex
     * associated with this clique.
     */
    id_t make_cover(const domain_type& domain) {
      id_t v = jt_.find_cluster_cover(domain);
      if (v) {
        return v;
      } else {
        retriangulate(iterator_range<const domain_type*>(&domain, &domain+1));
        return jt_.find_cluster_cover(domain);
      }
    }

    /**
     * Merges two vertices in the junction tree. This operation
     * swings all edges from the source of the supplied edge to the
     * target. The source is removed from the graph.
     */
    id_t merge(const edge_type& e) {
      id_t u = e.source();
      id_t v = e.target();

      // compute the marginal for the new clique clique(u) | clique(v)
      F marginal;
      if (superset(clique(u), clique(v))) {
        marginal = std::move(jt_[u]);
      } else {
        marginal = jt_[u] * jt_[v];
        marginal /= jt_[e];
      }

      // merge the edge and set the new marginal
      jt_.merge(e);
      jt_[v] = std::move(marginal);
      return v;
    }

    /**
     * Removes a vertex from the junction tree if its clique is nonmaximal.
     * \return the vertex merged to or the null vertex if not merged
     */
    id_t remove_if_nonmaximal(id_t u) {
      for (edge_type e : out_edges(u)) {
        if (subset(clique(u), clique(e.target()))) {
          return merge(e);
        }
      }
      return id_t();
    }

    // Distribution updates
    //==========================================================================

    /**
     * Multiplies the supplied collection of factors into this
     * decomposable model and renormalizes it.
     *
     * \tparam Range A forward range over factors that can be multiplied to F.
     */
    template <typename Range>
    decomposable& operator*=(const Range& factors) {
      // Retriangulate the model so that it contains a clique for each factor.
      std::vector<std::reference_wrapper<const domain_type> > domains;
      for (const auto& factor : factors) {
        domains.push_back(factor.arguments());
      }
      retriangulate(domains);

      // For each factor, multiply it into a clique that subsumes it.
      for (const auto& factor : factors) {
        if (!factor.arguments().empty()) {
          id_t v = jt_.find_cluster_cover(factor.arguments());
          assert(v);
          jt_[v] *= factor;
        }
      }

      // Recalibrate and renormalize the model.
      calibrate();
      normalize();
      return *this;
    }

    /**
     * Multiplies the supplied factor into this decomposable model and
     * renormalizes the model.
     */
    decomposable& operator*=(const F& factor) {
      id_t v = make_cover(factor.arguments());
      jt_[v] *= factor;
      distribute_evidence(v);
      return *this;
    }

    /**
     * Conditions this decomposable model on an assignment to one or
     * more of its variables and returns the likelihood of the evidence.
     * \todo compute the likelihood of evidence, reconnect the tree
     */
    result_type condition(const assignment_type& a) {
      domain_type restricted = restricted_args(a);

      // Update each affected clique
      jt_.intersecting_clusters(restricted, [&](id_t v) {
          F& factor = jt_[v];
          factor = factor.restrict(a);
          if (factor.arguments().empty()) {
            jt_.remove_vertex(v);
          } else {
            jt_.update_cluster(v, factor.arguments());
          }
        });

      // Update each affected separator
      jt_.intersecting_separators(restricted, [&](const edge_type& e) {
          F& factor = jt_[e];
          factor = factor.restrict(a);
          jt_.update_separator(e, factor.arguments());
        });

      // Update the arguments & recalibrate.
      calibrate();
      normalize();
      return result_type(1);
    }

    /**
     * Conditions the decomposable model and returns the result as a factor.
     */
    F condition_flatten(const assignment_type& a) const {
      F result(typename F::result_type(1));
      for (id_t v : jt_.vertices()) {
        result *= jt_[v].restrict(a);
      }
      for (edge_type e : jt_.edges()) {
        result /= jt_[e].restrict(a);
      }
      return result;
    }

    // Iterators
    //==========================================================================
    /**
     * An iterator over the factors of a decomposable model.
     * For a clique marginal, returns a reference to the factor.
     * For a separator marginal, returns a reference to a temporary
     * that holds the inverted marginal.
     */
    class const_iterator
      : public std::iterator<std::forward_iterator_tag, const F> {
    public:
      //! end constructor
      const_iterator()
        : dm_(nullptr), remaining_(0) { }

      //! begin constructor
      explicit const_iterator(const decomposable* dm)
        : dm_(dm),
          vit_(dm->vertices().begin()),
          eit_(dm->edges().begin()),
          remaining_(dm->num_vertices() + dm->num_edges()) { }

      const F& operator*() const {
        assert(remaining_ > 0);
        if (remaining_ > dm_->num_edges()) {
          return (*dm_)[*vit_];
        } else {
          return inv_potential_;
        }
      }

      const_iterator& operator++() {
        if (remaining_ > dm_->num_edges()) {
          ++vit_;
        } else if (remaining_ > 0) {
          ++eit_;
        }
        --remaining_;
        if (remaining_ > 0 && remaining_ <= dm_->num_edges()) {
          inv_potential_ = typename F::result_type(1) / (*dm_)[*eit_];
        }
        return *this;
      }

      const_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error(
          "decomposable::const_iterator does not support postincrement");
      }

      bool operator==(const const_iterator& other) const {
        return remaining_ == other.remaining_;
      }

      bool operator!=(const const_iterator& other) const {
        return remaining_ != other.remaining_;
      }

    private:
      //! A pointer to the decomposable model being iterated over.
      const decomposable* dm_; //!< decomposable model iterated over
      vertex_iterator vit_;    //!< current vertex iterator.
      edge_iterator eit_;      //!< current edge iterator.
      std::ptrdiff_t remaining_;    //!< number of factors left (incl. current one)
      F inv_potential_;        //!< temporary holding the inverted potential

    }; // class potential_iterator

    // Private members
    //==========================================================================
  private:
    /**
     * Passes flows outwards from the supplied vertex.
     */
    void distribute_evidence(id_t v) {
      pre_order_traversal(jt_, v, [&](const edge_type& e) {
          jt_[e.target()] /= jt_[e];
          jt_[e.source()].marginal(separator(e), jt_[e]);
          jt_[e.target()] *= jt_[e];
        });
    }

    /**
     * Recalibrates the model by passing flows using the message
     * passing protocol.
     */
    void calibrate() {
      mpp_traversal(jt_, id_t(), [&](const edge_type& e) {
          jt_[e.target()] /= jt_[e];
          jt_[e.source()].marginal(separator(e), jt_[e]);
          jt_[e.target()] *= jt_[e];
        });
    }

    /**
     * Normalizes this decomposable model; all clique and separator
     * marginals are normalized.
     */
    void normalize() {
      for (id_t v : vertices()) { jt_[v].normalize(); }
      for (edge_type e : edges()) { jt_[e].normalize(); }
    }

    /**
     * Returns the arguments of this model that are restricted by an
     * assignment.
     */
    domain_type restricted_args(const assignment_type& a) const {
      domain_type result;
      for (variable_type v : arguments()) {
        if (a.count(v)) {
          result.push_back(v);
        }
      }
      return result;
    }

    /**
     * Returns the arguments in this model intersecting the given domain.
     */
    domain_type intersecting_args(const domain_type& dom) const {
      domain_type result;
      for (variable_type v : dom) {
        if (jt_.count(v)) {
          result.push_back(v);
        }
      }
      return result;
    }

    //! The underlying junction tree
    graph_type jt_;

  }; // class decomposable

} // namespace libgm

namespace boost {

  //! A traits class that lets decomposable_model work in BGL algorithms
  template <typename F>
  struct graph_traits< libgm::decomposable<F> >
    : public graph_traits<
        libgm::cluster_graph<typename F::variable_type, F, F> > { };

} // namespace boost

#endif
