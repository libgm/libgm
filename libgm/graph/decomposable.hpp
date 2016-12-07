#ifndef LIBGM_DECOMPOSABLE_HPP
#define LIBGM_DECOMPOSABLE_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/argument/assignment.hpp>
#include <libgm/factor/utility/operations.hpp>
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
   * \tparam Arg
   *         A type that represents an individual argument (node).
   * \tparam F
   *         A type representing the factors. The type must support
   *         multiplication and division operations.
   *
   * \ingroup model
   */
  template <typename Arg, typename F>
  class decomposable {

    using graph_type = cluster_graph<Arg, F, F>;

    // Public type declarations
    //--------------------------------------------------------------------------
  public:
    // Vertex type, edge type, and properties
    using argument_type   = Arg;
    using vertex_type     = id_t;
    using edge_type       = undirected_edge<id_t>;
    using vertex_property = F;
    using edge_property   = F;

    // Iterators
    using argument_iterator = typename graph_type::argument_iterator;
    using vertex_iterator   = typename graph_type::vertex_iterator;
    using neighbor_iterator = typename graph_type::neighbor_iterator;
    using edge_iterator     = typename graph_type::edge_iterator;
    using in_edge_iterator  = typename graph_type::in_edge_iterator;
    using out_edge_iterator = typename graph_type::out_edge_iterator;
    class iterator; // forward declaration

    // Factor types
    using real_type   = typename F::real_type;
    using result_type = typename F::result_type;
    using factor_type = F;

    // Constructors and destructors
    //--------------------------------------------------------------------------
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
    //--------------------------------------------------------------------------

    //! Returns the range of arguments of the model.
    iterator_range<argument_iterator> arguments() const {
      return jt_.arguments();
    }

    //! Returns the range of all vertices (clique ids) of the model.
    iterator_range<vertex_iterator>
    vertices() const {
      return jt_.vertices();
    }

    //! Returns all edges in the graph.
    iterator_range<edge_iterator>
    edges() const {
      return jt_.edges();
    }

    //! Returns the vertices (clique ids) adjacent to u.
    iterator_range<neighbor_iterator>
    neighbors(id_t u) const {
      return jt_.neighbors(u);
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
    bool contains(undirected_edge<id_t> e) const {
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

    //! Returns the first vertex or the null vertex if the graph is empty.
    id_t root() const {
      return jt_.empty() ? id_t() : jt_.vertices().front();
    }

    //! Returns true if the graph has no vertices / no arguments.
    bool empty() const {
      return jt_.empty();
    }

    //! Returns the number of arguments in the model.
    std::size_t num_arguments() const {
      return jt_.num_arguments();
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
    undirected_edge<id_t> reverse(undirected_edge<id_t> e) const {
      return e.reverse();
    }

    //! Returns the clique associated with a vertex.
    const domain<Arg>& clique(id_t v) const {
      return jt_.cluster(v);
    }

    //! Returns the separator associated with an edge.
    const domain<Arg>& separator(undirected_edge<id_t> e) const {
      return jt_.separator(e);
    }

    //! Returns the index mapping from a domain to the given clique.
    uint_vector index(id_t v, const domain<Arg>& dom) const {
      return jt_.index(v, dom);
    }

    //! Returns the index mapping from a domain to the given separator.
    uint_vector index(edge_type e, const domain<Arg>& dom) const {
      return jt_.index(e, dom);
    }

    //! Returns the index mapping from the separator to the source clique.
    const uint_vector& source_index(undirected_edge<id_t> e) const {
      return jt_.source_index(e);
    }

    //! Returns the index mapping from the separator to the target clique.
    const uint_vector& target_index(undirected_edge<id_t> e) const {
      return jt_.target_index(e);
    }

    //! Returns the marginal associated with a vertex.
    const F& operator[](id_t u) const {
      return jt_[u];
    }

    //! Returns the marginal associated with an edge.
    const F& operator[](undirected_edge<id_t> e) const {
      return jt_[e];
    }

    //! Return the annotated marginal associated with a vertex.
    const annotated<Arg, F>& property(id_t u) const {
      return jt_.property(u);
    }

    //! Returns the annotated marginal associated with an edge.
    const annotated<Arg, F>& property(undirected_edge<id_t> e) const {
      return jt_.property(e);
    }

    //! Returns the iterator to the first factor.
    iterator begin() const {
      return iterator(this);
    }

    //! Returns the iterator to the one past the last factor.
    iterator end() const {
      return iterator();
    }


    //! Returns the underlying junction tree.
    const graph_type& jt() const {
      return jt_;
    }

    // Queries
    //--------------------------------------------------------------------------

    /**
     * Computes the Markov graph capturing the dependencies in this model.
     */
    void markov_graph(undirected_graph<Arg>& mg) const {
      for (id_t v : vertices()) {
        mg.make_clique(clique(v));
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
        if (msg) {
          *msg = "The underlying graph is not a tree";
        }
        return false;
      }
      if (!jt_.running_intersection()) {
        if (msg) {
          *msg = "The underlying graph does not satisfiy RIP";
        }
        return false;
      }
      for (id_t v : vertices()) {
        if (jt_[v].shape() != F::param_shape(clique(v))) {
          if (msg) {
            *msg = "Inconsistent shape for clique " + clique(v).str();
          }
          return false;
        }
      }
      for (edge_type e : edges()) {
        if (jt_[e].shape() != F::param_shape(separator(e))) {
          if (msg) {
            *msg = "Inconsistent shape for separator " + separator(e).str();
          }
          return false;
        }
      }
      return true;
    }

    /**
     * Computes a marginal over an arbitrary subset of arguments.
     * The arguments must be all present in this decomposble model.
     */
    F marginal(const domain<Arg>& dom) const {
      if (dom.empty()) {
        return F(result_type(1));
      }

      // Look for a separator that covers the arguments.
      edge_type e = jt_.find_separator_cover(dom);
      if (e) {
        return jt_[e].marginal(index(e, dom));
      }

      // Look for a clique that covers the arguments.
      id_t v = jt_.find_cluster_cover(dom);
      if (v) {
        return jt_[v].marginal(index(v, dom));
      }

      // Otherwise, compute the factors whose product represents
      // the marginal
      std::list<std::pair<domain<Arg>, F> > factors;
      marginal(domain, factors);
      annotated<Arg, F> product = sum_product<Arg, F>().combine_all(factors);
      return product.object.marginal(product.domain.index(domain));
    }

    /**
     * Computes a list of factors whose product represents
     * a marginal over a subset of arguments.
     */
    void marginal(const domain<Arg>& dom,
                  std::list<annotated<Arg, F> >& factors) const {
      factors.clear();
      if (domain.empty()) return;

      const_cast<graph_type&>(jt_).mark_subtree_cover(dom, false);
      for (id_t v : vertices()) {
        if (jt_.marked(v)) {
          factors.emplace_back(clique(v), jt_[v]);
        }
      }
      for (edge_type e : edges()) {
        if (jt_.marked(e)) {
          factors.emplace_back(separator(e), result_type(1) / jt_[e]);
        }
      }

      variable_elimination(factors, dom, sum_product<Arg, F>());
    }

    /**
     * Computes a decomposable model that represents the marginal
     * distribution over one ore more arguments.
     * Note: This operation can create large cliques.
     */
    void marginal(const domain<Arg>& domain, decomposable& result) const {
      std::list<std::pair<domain<Arg>, F> > factors;
      marginal(domain, factors);
      result.reset(factors);
    }

    /**
     * Computes the entropy of the distribution represented by this
     * decomposable model.
     */
    real_type entropy() const {
      real_type result(0);
      for (id_t v : vertices()) {
        result += jt_[v].entropy();
      }
      for (edge_type e : edges()) {
        result -= jt_[e].entropy();
      }
      return result;
    }

    /**
     * Computes the entropy over a subset of arguments.
     */
    real_type entropy(const domain<Arg>& domain) const {
      // first try to compute the entropy directly from the marginals
      edge_type e = jt_.find_separator_cover(dom);
      if (e) {
        return jt_[e].entropy(index(e, dom));
      }

      id_t v = jt_.find_cluster_cover(domain);
      if (v) {
        return jt_[v].entropy(index(v, dom));
      }

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
    real_type conditional_entropy(const domain<Arg>& y,
                                  const domain<Arg>& x) const {
      return entropy(x + y) - entropy(x);
    }

    /**
     * Computes the mutual information I(A ; B) between two subsets of*
     * arguments of this model.
     */
    real_type mutual_information(const domain<Arg>& a,
                                 const domain<Arg>& b) const {
      return entropy(a) + entropy(b) - entropy(a + b);
    }

    /**
     * Computes the conditional mutual information I(A; B | C),
     * where A,B,C must be subsets of the arguments of this model.
     * This is computed using I(A; B | C) = H(A | C) - H(A | B, C).
     *
     * @param base   Base of logarithm.
     * @return double representing the conditional mutual information.
     */
    real_type conditional_mutual_information(const domain<Arg>& a,
                                             const domain<Arg>& b,
                                             const domain<Arg>& c) const {
      return conditional_entropy(a, c) - conditional_entropy(a, b + c);
    }

    /**
     * Compute the maximum probability and stores the corresponding
     * assignment to a.
     */
    result_type maximum(assignment<Arg, real_type>& a) const {
      a.clear();
      if (empty()) {
        return result_type(1);
      }

      // copy the clique marginals into factors
      std::unordered_map<id_t, F> factor;
      for (id_t v : vertices()) {
        factor[v] = jt_[v];
      }

      // collect evidence
      post_order_traversal(jt_, root(), [&](const edge_type& e) {
          F& f = factor[e.target()];
          f.dims(target_index(e)) *= factor[e.source()].maximum(source_index(e));
          f.dims(target_index(e)) /= jt_[e];
        });

      // extract the maximum for the root clique
      a.insert_or_assign(jt.clique(root()), factor[root()].arg_max());

      // distribute evidence
      pre_order_traversal(jt_, root(), [&](const edge_type& e) {
          F f = factor[e.target()).restrict(target_index(e),
                                            a.values(separator(e)));
          a.insert_or_assign(clique(e.target()) - separator(e), f.arg_max());
        });

      return factor[root()].max();
    }

    /**
     * Draws a random sample from this model.
     * \tparam Generator a random number generator.
     */
    template <typename Generator>
    void sample(Generator& rng, assignment<Arg, real_type>& a) const {
      a.clear();
      a.insert_or_assign(clique(root()), jt_[root()].sample(rng));
      pre_order_traversal(jt_, root(), [&](const edge_type& e) {
          F f = jt_[e.target()].restrict(target_index(e),
                                         a.values(separator(e)));
          a.insert_or_assign(clique(e.target()) - separator(e), f.sample(rng));
        });
    }

    /**
     * Returns the completel log-likelihood of the given assignment.
     */
    real_type log(const assignment<Arg, real_type>& a) const {
      real_type result(0);
      for (id_t v : vertices()) {
        result += jt_[v].log(a.values(clique(v)));
      }
      for (edge_type e : edges()) {
        result -= jt_[e].log(a.values(separator(e)));
      }
      return result;
    }

    /**
     * Returns the probability of the assignment.
     * if the assignment includes all the arguments of this model,
     * this function computes the joint probaiblity p(a).
     * Otherwise, this function computes the marginal probability.
     */
    result_type operator()(const assignment<Arg, real_type>& a) const {
      return std::exp(log(a));
    }

    // Restructuring operations
    //--------------------------------------------------------------------------

    //! Clears all factors and arguments from this model.
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
     *
     * \tparam Range
     *         A single pass range with elements convertible to value_type.
     */
    template <typename Range>
    void reset_marginals(const Range& marginals) {
      clear();

      // initialize the clique marginals and the tree structure
      for (const annotated<Arg, F>& marginal : marginals) {
        jt_.add_cluster(marginal.domain, marginal.object);
      }
      jt_.mst_edges();

      // compute the separator marginals
      for (edge_type e : edges()) {
        if (clique(e.source()).size() < clique(e.target()).size()) {
          jt_[e] = jt_[e.source()].marginal(source_index(e));
        } else {
          jt_[e] = jt_[e.target()].marginal(target_index(e));
        }
      }

      // calibrate & normalize in case the marginals are not consistent
      calibrate();
      normalize();
    }

    /**
     * Initializes the decomposable model to a single marginal.
     */
    void reset_marginal(const domain<Arg>& dom, const F& factor) {
      clear();
      jt_.add_cluster(dom, factor);
    }

    /**
     * Restructures this decomposable model so that it includes the
     * supplied cliques. These cliques can include new arguments
     * (which are not present in this model). In this case, the
     * marginals over these new arguments will be set to 1.
     *
     * \tparam Range
     *         A single pass range with elements convertible to domain<Arg>.
     *
     * \todo right now we retriangulate the entire model, but only
     *       the subtree containing the parameter vars must be
     *       retriangulated.
     */
    template <typename Range>
    void retriangulate(const Range& cliques) {
      // Create a graph with the cliques of this decomposable model.
      undirected_graph<Arg> mg;
      markov_graph(mg);

      // Add the new cliques
      for (const domain<Arg>& clique : cliques) {
        mg.make_clique(clique);
      }

      // Now create a new junction tree for the Markov graph and
      // initialize the clique/separator marginals
      graph_type jt;
      jt.triangulated(mg, min_degree_strategy());
      for (id_t v : jt.vertices()) {
        jt[v] = marginal(jt.cluster(v));
      }
      for (edge_type e : jt.edges()) {
        jt[e] = marginal(jt.separator(e));
      }

      // Swap in the new junction tree
      swap(jt, jt_);
    }

    /**
     * Restructures this decomposable model so that it has a clique
     * that covers the supplied arguments, and returns the vertex
     * associated with this clique.
     */
    id_t make_cover(const domain<Arg>& dom) {
      id_t v = jt_.find_cluster_cover(dom);
      if (v) {
        return v;
      } else {
        retriangulate(make_singleton_range(dom));
        return jt_.find_cluster_cover(dom);
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

      // compute the marginal for the new clique clique(u) + clique(v)
      F marginal;
      if (superset(clique(u), clique(v))) {
        marginal = std::move(jt_[u]);
      } else {
        marginal = jt_[u].dims(source_index(e)) * jt_[v].dims(target_index(e));
        marginal.dims(????.index(separator(e))) /= jt_[e];
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
    //--------------------------------------------------------------------------

    /**
     * Multiplies the supplied collection of factors into this
     * decomposable model and renormalizes it.
     *
     * \tparam Range A forward range over factors that can be multiplied to F.
     */
    template <typename Range>
    decomposable& multiply_in(const Range& factors) {
      retriangulate(make_domain_range(factor));

      // For each factor, multiply it into a clique that subsumes it.
      for (const annotated<Arg, F>& factor : factors) {
        const domain<Arg>& dom = factor.domain;
        if (!dom.empty()) {
          id_t v = jt_.find_cluster_cover(dom);
          assert(v);
          jt_[v].dims(index(v, dom)) *= factor.object;
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
    decomposable& multiply_in(const domain<Arg>& dom, const F& factor) {
      id_t v = make_cover(dom);
      jt_[v].dims(index(v, dom)) *= factor;
      distribute_evidence(v);
      return *this;
    }

    /**
     * Conditions this decomposable model on an assignment to one or
     * more of its arguments and returns the likelihood of the evidence.
     * \todo compute the likelihood of evidence, reconnect the tree
     */
    result_type condition(const assignment<Arg, real_type>& a) {
      domain<Arg> restricted = restricted_args(a);

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
    annotated<Arg, F>
    condition_flatten(const assignment<Arg, real_type>& a) const {
      std::list<annotated<Arg, F> > factors;
      for (id_t v : jt_.vertices()) {
        factors.emplace_back(clique(v) - a, ...); // TOOD
        result *= jt_[v].restrict(a);
      }
      for (edge_type e : jt_.edges()) {
        result /= jt_[e].restrict(a);
      }
      return result;
    }

    // Iterators
    //--------------------------------------------------------------------------

    /**
     * An iterator over the factors of a decomposable model.
     * For a clique marginal, returns a reference to the factor.
     * For a separator marginal, returns a reference to a temporary
     * that holds the inverted marginal.
     */
    class iterator
      : public std::iterator<std::forward_iterator_tag,
                             const annotated<Arg, F> > {
    public:
      //! end constructor
      iterator()
        : dm_(nullptr), remaining_(0) { }

      //! begin constructor
      explicit iterator(const decomposable* dm)
        : dm_(dm),
          vit_(dm->vertices().begin()),
          eit_(dm->edges().begin()),
          remaining_(dm->num_vertices() + dm->num_edges()) { }

      const annotated<Arg, F>& operator*() const {
        assert(remaining_ > 0);
        if (remaining_ > dm_->num_edges()) {
          return dm_.property(*vi);
        } else {
          return inv_potential_;
        }
      }

      iterator& operator++() {
        if (remaining_ > dm_->num_edges()) {
          ++vit_;
        } else if (remaining_ > 0) {
          ++eit_;
        } else {
          throw std::logic_error("Attempt to iterate past end");
        }
        --remaining_;
        if (remaining_ > 0 && remaining_ <= dm_->num_edges()) {
          inv_potential_.domain = dm_.separator(*eit_);
          inv_potential_.object = typename F::result_type(1) / (*dm_)[*eit_];
        }
        return *this;
      }

      iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error(
          "decomposable::const_iterator does not support postincrement");
      }

      bool operator==(const iterator& other) const {
        return remaining_ == other.remaining_;
      }

      bool operator!=(const iterator& other) const {
        return remaining_ != other.remaining_;
      }

    private:
      //! A pointer to the decomposable model being iterated over.
      const decomposable* dm_; //!< decomposable model iterated over
      vertex_iterator vit_;    //!< current vertex iterator.
      edge_iterator eit_;      //!< current edge iterator.
      std::size_t remaining_;  //!< number of factors left (incl. current one)
      annotated<Arg, F> inv_potential_; //!< temporary for inverted potential

    }; // class iterator

    // Private members
    //--------------------------------------------------------------------------
  private:

    /**
     * Passes the flow along an edge.
     */
    void pass_flow(edge_type e) {
      jt_[e.target()].dims(target_index(e)) /= jt_[e];
      jt_[e] = jt_[e.source()].marginal(source_index(e));
      jt_[e.target()].dims(target_index(e)) *= jt_[e];
    }

    /**
     * Passes flows outwards from the supplied vertex.
     */
    void distribute_evidence(id_t v) {
      pre_order_traversal(jt_, v, [&](const edge_type& e) { pass_flow(e); });
    }

    /**
     * Recalibrates the model by passing flows using the message
     * passing protocol.
     */
    void calibrate() {
      mpp_traversal(jt_, id_t(), [&](const edge_type& e) { pass_flow(e); });
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
    domain<Arg> restricted_args(const assignment<Arg, real_type>& a) const {
      domain<Arg> result;
      for (Arg arg : arguments()) {
        if (a.count(arg)) {
          result.push_back(arg);
        }
      }
      return result;
    }

    /**
     * Returns the arguments in this model intersecting the given domain.
     */
    domain<Arg> intersecting_args(const domain<Arg>& dom) const {
      domain<Arg> result;
      for (Arg arg : dom) {
        if (jt_.count(arg)) {
          result.push_back(arg);
        }
      }
      return result;
    }

    //! The underlying junction tree
    graph_type jt_;

  }; // class decomposable

} // namespace libgm

#endif
