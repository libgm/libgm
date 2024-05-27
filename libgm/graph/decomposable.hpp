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
 * A Decomposable representation of a probability distribution.
 * Conceptually, Decomposable model is a junction tree, in which
 * each vertex and each edge is associated with a factor.
 * The distribution is equal to to the product of clique marginals,
 * divided by the product of separator marginals.
 *
 * \tparam F
 *         A type representing the factors. The type must support
 *         multiplication and division operations.
 *
 * \ingroup model
 */
class Decomposable {
  // Public type declarations
  //--------------------------------------------------------------------------

public:
  // Vertex type, edge type, and properties
  using vertex_descriptor = ClusterGraph::vertex_descriptor;
  using edge_descriptor   = ClusterGraph::edge_descriptor

  // Iterators
  using argument_iterator  = typename graph_type::argument_iterator;
  using out_edge_iterator  = typename graph_type::out_edge_iterator;
  using in_edge_iterator   = typename graph_type::in_edge_iterator;
  using adjacency_iterator = typename graph_type::neighbor_iterator;
  using vertex_iterator    = typename graph_type::vertex_iterator;
  using edge_iterator      = typename graph_type::edge_iterator;
  class iterator; // forward declaration

  struct Edge : BaseGraph::Edge {
    Bidirectional<Dims> index;


  }

  // // Factor types
  // using real_type   = typename F::real_type;
  // using result_type = typename F::result_type;
  // using factor_type = F;

  // Constructors and destructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. The distribution has no arguments and
   * is identically one.
   */
  Decomposable() { }

  //! Swaps two Decomposable models in place.
  friend void swap(Decomposable& a, Decomposable& b) {
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

  //! Returns true if the two Decomposable models are identical.
  friend bool operator==(const Decomposable& a, const Decomposable& b) {
    return a.jt_ == b.jt_;
  }

  //! Returns true if the two Decomposable models are not identical.
  friend bool operator!=(const Decomposable& a, const Decomposable& b) {
    return a.jt_ != b.jt_;
  }

  //! Prints the Decomposable model to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const Decomposable& dm) {
    out << dm.jt_;
    return out;
  }

  // Accessors
  //--------------------------------------------------------------------------

  //! Returns the range of arguments of the model.
  boost::iterator_range<argument_iterator> arguments() const {
    return jt_.arguments();
  }

  //! Returns true if the graph contains the given vertex.
  bool contains(Vertex* u) const {
    return jt_.contains(u);
  }

  //! Returns true if the graph contains an undirected edge {u, v}.
  bool contains(Vertex* u, Vertex* v) const {
    return jt_.contains(u, v);
  }

  //! Returns true if the graph contains an undirected edge.
  bool contains(edge_descriptor e) const {
    return jt_.contains(e);
  }

  //! Returns the first vertex or the null vertex if the graph is empty.
  Vertex* root() const {
    return jt_.empty() ? nullptr : jt_.vertices().front();
  }

  //! Returns true if the graph has no vertices / no arguments.
  bool empty() const {
    return jt_.empty();
  }

  //! Returns the number of arguments in the model.
  std::size_t num_arguments() const {
    return jt_.num_arguments();
  }

  //! Given an undirected edge (u, v), returns the equivalent edge (v, u).
  edge_descriptor reverse(edge_descriptor e) const {
    return e.reverse();
  }

  //! Returns the clique associated with a vertex.
  const Domain& clique(Vertex* v) const;

  //! Returns the separator associated with an edge.
  const Domain& separator(edge_descriptor e) const {
    return jt_.separator(e);
  }

  //! Returns the index mapping from a domain to the given clique.
  uint_vector index(Vertex* v, const Domain& dom) const {
    return jt_.index(v, dom);
  }

  //! Returns the index mapping from a domain to the given separator.
  uint_vector index(edge_descriptor e, const Domain& dom) const {
    return jt_.index(e, dom);
  }

  //! Returns the index mapping from the separator to the source clique.
  const uint_vector& source_index(edge_descriptor e) const {
    return e->index_(e.source(), e.target());
  }

  //! Returns the index mapping from the separator to the target clique.
  const uint_vector& target_index(edge_descriptor e) const {
    return e->index_(e.target(), e.source());
  }

  //! Returns the marginal associated with a vertex.
  const Object& operator[](Vertex* u) const {
    return u->property_;
  }

  //! Returns the marginal associated with an edge.
  const Object& operator[](edge_descriptor e) const {
    return e->property_;
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
  MarkovNetwork markov_network() const {
    MarkovNetwork mn;
    for (Vertex* v : vertices()) {
      mn.make_clique(v->cluster());
    }
  }

  /**
   * Returns tre if this Decomposable model is valid.
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
    for (Vertex* v : vertices()) {
      if (jt_[v].shape() != F::param_shape(clique(v))) {
        if (msg) {
          *msg = "Inconsistent shape for clique " + clique(v).str();
        }
        return false;
      }
    }
    for (edge_descriptor e : edges()) {
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
  Object marginal(const Domain& dom) const {
    if (dom.empty()) {
      return one;
    }

    // Look for a separator that covers the arguments.
    edge_descriptor e = jt_.find_separator_cover(dom);
    if (e) {
      return jt_[e].marginal(dims(e, dom));
    }

    // Look for a clique that covers the arguments.
    Vertex* v = jt_.find_cluster_cover(dom);
    if (v) {
      return jt_[v].marginal(dims(v, dom));
    }

    // Otherwise, compute the factors whose product represents
    // the marginal
    std::list<std::pair<Domain, F>> factors;
    marginal(domain, factors);

    annotated<Arg, F> product = sum_product<F>().combine_all(factors);
    return product.object.marginal(product.domain.index(domain));
  }

  /**
   * Computes a list of factors whose product represents
   * a marginal over a subset of arguments.
   */
  void marginal(const Domain& dom, std::list<annotated<Arg, F> >& factors) {
    factors.clear();
    if (domain.empty()) return;

    jt_.mark_subtree_cover(dom, false);
    for (Vertex* v : *this) {
      if (jt_.marked(v)) {
        factors.emplace_back(v->clique(), jt_[v]);
      }
    }
    for (edge_descriptor e : edges()) {
      if (jt_.marked(e)) {
        factors.emplace_back(e->separator(), one / jt_[e]);
      }
    }

    variable_elimination(factors, dom, sum_product<Arg, F>());
  }

  /**
   * Computes a Decomposable model that represents the marginal
   * distribution over one ore more arguments.
   * Note: This operation can create large cliques.
   */
  void marginal(const Domain& domain, Decomposable& result) const {
    std::list<std::pair<Domain, F> > factors;
    marginal(domain, factors);
    result.reset(factors);
  }

  /**
   * Computes the entropy of the distribution represented by this
   * Decomposable model.
   */
  real_type entropy() const {
    real_type result(0);
    for (Vertex* v : vertices()) {
      result += jt_[v].entropy();
    }
    for (edge_descriptor e : edges()) {
      result -= jt_[e].entropy();
    }
    return result;
  }

  /**
   * Computes the entropy over a subset of arguments.
   */
  real_type entropy(const Domain& domain) const {
    // first try to compute the entropy directly from the marginals
    edge_descriptor e = jt_.find_separator_cover(dom);
    if (e) {
      return jt_[e].entropy(index(e, dom));
    }

    Vertex* v = jt_.find_cluster_cover(domain);
    if (v) {
      return jt_[v].entropy(index(v, dom));
    }

    // failing that, compute the marginal of the model
    Decomposable tmp;
    marginal(domain, tmp);
    return tmp.entropy();
  }

  /**
   * Computes the conditional entropy H(Y | X), where Y, X are subsets
   * of the arguments of this model.
   * \todo see if we can optimize this
   */
  real_type conditional_entropy(const Domain& y,
                                const Domain& x) const {
    return entropy(x + y) - entropy(x);
  }

  /**
   * Computes the mutual information I(A ; B) between two subsets of*
   * arguments of this model.
   */
  real_type mutual_information(const Domain& a,
                                const Domain& b) const {
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
  real_type conditional_mutual_information(const Domain& a,
                                            const Domain& b,
                                            const Domain& c) const {
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
    std::unordered_map<Vertex*, F> factor;
    for (Vertex* v : vertices()) {
      factor[v] = jt_[v];
    }

    // collect evidence
    post_order_traversal(jt_, root(), [&](const edge_descriptor& e) {
        F& f = factor[e.target()];
        f.dims(target_index(e)) *= factor[e.source()].maximum(source_index(e));
        f.dims(target_index(e)) /= jt_[e];
      });

    // extract the maximum for the root clique
    a.insert_or_assign(jt.clique(root()), factor[root()].arg_max());

    // distribute evidence
    pre_order_traversal(jt_, root(), [&](const edge_descriptor& e) {
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
    pre_order_traversal(jt_, root(), [&](const edge_descriptor& e) {
        F f = jt_[e.target()].restrict(target_index(e),
                                        a.values(separator(e)));
        a.insert_or_assign(clique(e.target()) - separator(e), f.sample(rng));
      });
  }

  /**
   * Returns the completel log-likelihood of the given assignment.
   */
  real_type log(const Assignment& a) const {
    LogProbability result();
    for (Vertex* v : vertices()) {
      result += jt_[v].log(a.values(v->clique()));
    }
    for (edge_descriptor e : edges()) {
      result -= jt_[e].log(a.values(e->separator()));
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
   * Initializes the Decomposable model to the product of the given
   * factors.
   * \tparam Range a single pass range with elements convertible to F
   */
  template <typename Range>
  void reset(const Range& factors) {
    clear();
    operator*=(factors);
  }

  /**
   * Initializes the Decomposable model to the given range of marginals.
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
    for (edge_descriptor e : edges()) {
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
   * Initializes the Decomposable model to a single marginal.
   */
  void reset_marginal(const Domain& dom, const F& factor) {
    clear();
    jt_.add_cluster(dom, factor);
  }

  /**
   * Restructures this Decomposable model so that it includes the
   * supplied cliques. These cliques can include new arguments
   * (which are not present in this model). In this case, the
   * marginals over these new arguments will be set to 1.
   *
   * \tparam Range
   *         A single pass range with elements convertible to Domain.
   *
   * \todo right now we retriangulate the entire model, but only
   *       the subtree containing the parameter vars must be
   *       retriangulated.
   */
  template <typename Range>
  void retriangulate(const Range& cliques) {
    // Create a graph with the cliques of this Decomposable model.
    undirected_graph<Arg> mg;
    markov_graph(mg);

    // Add the new cliques
    for (const Domain& clique : cliques) {
      mg.make_clique(clique);
    }

    // Now create a new junction tree for the Markov graph and
    // initialize the clique/separator marginals
    graph_type jt;
    jt.triangulated(mg, min_degree_strategy());
    for (Vertex* v : jt.vertices()) {
      jt[v] = marginal(jt.cluster(v));
    }
    for (edge_descriptor e : jt.edges()) {
      jt[e] = marginal(jt.separator(e));
    }

    // Swap in the new junction tree
    swap(jt, jt_);
  }

  /**
   * Restructures this Decomposable model so that it has a clique
   * that covers the supplied arguments, and returns the vertex
   * associated with this clique.
   */
  Vertex* make_cover(const Domain& dom) {
    Vertex* v = jt_.find_cluster_cover(dom);
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
  Vertex* merge(const edge_descriptor& e) {
    Vertex* u = e.source();
    Vertex* v = e.target();

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
  Vertex* remove_if_nonmaximal(Vertex* u) {
    for (edge_descriptor e : out_edges(u)) {
      if (subset(clique(u), clique(e.target()))) {
        return merge(e);
      }
    }
    return Vertex*();
  }

  // Distribution updates
  //--------------------------------------------------------------------------

  /**
   * Multiplies the supplied collection of factors into this
   * Decomposable model and renormalizes it.
   *
   * \tparam Range A forward range over factors that can be multiplied to F.
   */
  template <typename Range>
  Decomposable& multiply_in(const Range& factors) {
    retriangulate(make_domain_range(factor));

    // For each factor, multiply it into a clique that subsumes it.
    for (const annotated<Arg, F>& factor : factors) {
      const Domain& dom = factor.domain;
      if (!dom.empty()) {
        Vertex* v = jt_.find_cluster_cover(dom);
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
   * Multiplies the supplied factor into this Decomposable model and
   * renormalizes the model.
   */
  Decomposable& multiply_in(const Domain& dom, const F& factor) {
    Vertex* v = make_cover(dom);
    jt_[v].dims(index(v, dom)) *= factor;
    distribute_evidence(v);
    return *this;
  }

  /**
   * Conditions this Decomposable model on an assignment to one or
   * more of its arguments and returns the likelihood of the evidence.
   * \todo compute the likelihood of evidence, reconnect the tree
   */
  result_type condition(const assignment<Arg, real_type>& a) {
    Domain restricted = restricted_args(a);

    // Update each affected clique
    jt_.intersecting_clusters(restricted, [&](Vertex* v) {
        F& factor = jt_[v];
        factor = factor.restrict(a);
        if (factor.arguments().empty()) {
          jt_.remove_vertex(v);
        } else {
          jt_.update_cluster(v, factor.arguments());
        }
      });

    // Update each affected separator
    jt_.intersecting_separators(restricted, [&](const edge_descriptor& e) {
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
   * Conditions the Decomposable model and returns the result as a factor.
   */
  annotated<Arg, F>
  condition_flatten(const assignment<Arg, real_type>& a) const {
    std::list<annotated<Arg, F> > factors;
    for (Vertex* v : jt_.vertices()) {
      factors.emplace_back(clique(v) - a, ...); // TOOD
      result *= jt_[v].restrict(a);
    }
    for (edge_descriptor e : jt_.edges()) {
      result /= jt_[e].restrict(a);
    }
    return result;
  }

  // Iterators
  //--------------------------------------------------------------------------

  /**
   * An iterator over the factors of a Decomposable model.
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
    explicit iterator(const Decomposable* dm)
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
        "Decomposable::const_iterator does not support postincrement");
    }

    bool operator==(const iterator& other) const {
      return remaining_ == other.remaining_;
    }

    bool operator!=(const iterator& other) const {
      return remaining_ != other.remaining_;
    }

  private:
    //! A pointer to the Decomposable model being iterated over.
    const Decomposable* dm_; //!< Decomposable model iterated over
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
  void pass_flow(edge_descriptor e) {
    jt_[e.target()].dims(target_index(e)) /= jt_[e];
    jt_[e] = jt_[e.source()].marginal(source_index(e));
    jt_[e.target()].dims(target_index(e)) *= jt_[e];
  }

  /**
   * Passes flows outwards from the supplied vertex.
   */
  void distribute_evidence(Vertex* v) {
    pre_order_traversal(jt_, v, [&](const edge_descriptor& e) { pass_flow(e); });
  }

  /**
   * Recalibrates the model by passing flows using the message
   * passing protocol.
   */
  void calibrate() {
    mpp_traversal(jt_, Vertex*(), [&](const edge_descriptor& e) { pass_flow(e); });
  }

  /**
   * Normalizes this Decomposable model; all clique and separator
   * marginals are normalized.
   */
  void normalize() {
    for (Vertex* v : vertices()) { jt_[v].normalize(); }
    for (edge_descriptor e : edges()) { jt_[e].normalize(); }
  }

  /**
   * Returns the arguments of this model that are restricted by an
   * assignment.
   */
  Domain restricted_args(const assignment<Arg, real_type>& a) const {
    Domain result;
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
  Domain intersecting_args(const Domain& dom) const {
    Domain result;
    for (Arg arg : dom) {
      if (jt_.count(arg)) {
        result.push_back(arg);
      }
    }
    return result;
  }

  //! The underlying junction tree
  graph_type jt_;

}; // class Decomposable

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
  neighbors(Vertex* u) const {
    return jt_.neighbors(u);
  }

  //! Returns the edges incoming to a vertex.
  iterator_range<in_edge_iterator>
  in_edges(Vertex* u) const {
    return jt_.in_edges(u);
  }

  //! Returns the outgoing edges from a vertex.
  iterator_range<out_edge_iterator>
  out_edges(Vertex* u) const {
    return jt_.out_edges(u);
  }

  //! Returns an undirected edge (u, v). The edge must exist.
  edge_descriptor edge(Vertex* u, Vertex* v) const {
    return jt_.edge(u, v);
  }

  //! Returns the number of edges adjacent to a vertex.
  std::size_t in_degree(Vertex* u) const {
    return jt_.in_degree(u);
  }

  //! Returns the number of edges adjacent to a vertex.
  std::size_t out_degree(Vertex* u) const {
    return jt_.out_degree(u);
  }

  //! Returns the number of edges adjacent to a vertex.
  std::size_t degree(Vertex* u) const {
    return jt_.degree(u);
  }
  //! Returns the number of vertices.
  std::size_t num_vertices() const {
    return jt_.num_vertices();
  }

  //! Returns the number of edges.
  std::size_t num_edges() const {
    return jt_.num_edges();
  }



} // namespace libgm

#endif
