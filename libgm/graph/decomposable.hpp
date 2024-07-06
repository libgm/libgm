#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/argument/assignment.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/graph/cluster_graph.hpp>

namespace libgm {

/**
 * A Decomposable representation of a probability distribution.
 * Conceptually, Decomposable model is a junction tree, in which
 * each vertex and each edge is associated with a factor.
 * The distribution is equal to to the product of clique marginals,
 * divided by the product of separator marginals.
 *
 * \tparam R the real type (typically float or double)
 *
 * \ingroup model
 */
template <typename R>
class Decomposable : public ClusterGraph {
  // Public type declarations
  //--------------------------------------------------------------------------
public:
  // Factor interface.
  struct Potential : Factor<
    One<Potential>,
    MultiplyJoinIn<Potential, Potential>,
    DivideJoinIn<Potential, Potential>,
    Normalize<Potential>,
    Restrict<Potential, Vector>,
  > {};

  // Constructors and destructors
  //--------------------------------------------------------------------------
public:
  /**
   * Default constructor. The distribution has no arguments and is identically one.
   */
  Decomposable() = default;

  /// Swaps two Decomposable models in place.
  friend void swap(Decomposable& a, Decomposable& b) {
    std::swap(a.impl_, b.impl_);
    // TODO: vtables
  }

  // Queries
  //--------------------------------------------------------------------------
  /**
   * Returns true if this Decomposable model is valid.
   * Decomposable model is valid if the underlying junction tree satisfies
   * the running intersection property, and the cliques and separators
   * match the corresponding marginal.
   *
   * \param msg if not null, an object where the error message is stored
   */
  bool valid(std::string* msg = nullptr) const;

  /**
   * Computes a marginal over an arbitrary subset of arguments.
   * The arguments must be all present in this decomposble model.
   */
  Potential marginal(const Domain& dom) const;

  /**
   * Computes factors whose product represents a marginal over a subset of arguments.
   */
  void marginal(const Domain& dom, FactorGraph& fg);

  /**
   * Computes a decomposable model that represents the marginal distribution over
   * a subset of arguments. Note: This operation can create large cliques.
   */
  void marginal(const Domain& domain, Decomposable& result) const;

  /**
   * Compute the maximum probability and stores the corresponding assignment to a.
   */
  Potential maximum(Assignment& a) const;

  // Restructuring operations
  //--------------------------------------------------------------------------

  /// Clears all factors and arguments from this model.
  void clear();

  /**
   * Initializes the decomposable model to the product of the given
   * factors.
   */
  void reset(const FactorGraph& factors);

  /**
   * Initializes the decomposable model to the given clique marginals,
   */
  void reset(std::vector<Domain> domains, Object* data);

  /**
   * Initializes the decomposable model to a single marginal.
   */
  void reset(Domain dom, Potential marginal);

  /**
   * Initializes the decomposable model to the given markov network,
   * using the specified elimination streategy.
   */
  void triangulated(MarkovGraph& mg, const EliminationStrategy& strategy);

  /**
   * Restructures this decomposable model so that it includes the
   * supplied cliques. These cliques can include new arguments
   * (which are not present in this model). In this case, the
   * marginals over these new arguments will be set to 1.
   *
   * \todo right now we retriangulate the entire model, but only
   *       the subtree containing the parameter vars must be
   *       retriangulated.
   */
  void retriangulate(const std::vector<Domain>& cliques);

  /**
   * Restructures this Decomposable model so that it has a clique
   * that covers the supplied arguments, and returns the vertex
   * associated with this clique.
   */
  Vertex* make_cover(const Domain& dom);

  /**
   * Merges two vertices in the junction tree. This operation
   * swings all edges from the source of the supplied edge to the
   * target. The source is removed from the graph.
   */
  Vertex* merge(edge_descriptor e);

  /**
   * Removes a vertex from the junction tree if its clique is nonmaximal.
   * \return the vertex merged to or the null vertex if not merged
   */
  Vertex* remove_if_nonmaximal(Vertex* u);

  // Distribution updates
  //--------------------------------------------------------------------------

  /**
   * Multiplies the supplied collection of factors into this
   * Decomposable model and renormalizes it.
   */
  Decomposable& multiply_in(const std::vector<Domain>& domains, const Object* factors);

  /**
   * Multiplies the supplied factor into this Decomposable model and
   * renormalizes the model.
   */
  Decomposable& multiply_in(const Domain& dom, const Potential& factor);

  /**
   * Conditions this Decomposable model on an assignment to one or
   * more of its arguments and returns the likelihood of the evidence.
   * \todo compute the likelihood of evidence, reconnect the tree
   */
  void condition(const Assignemnt& a);

  /**
   * Conditions the Decomposable model and returns the result as a factor.
   */
  Annotated<Potential> condition_flatten(const Assignment& a) const;

}; // class Decomposable

template <typename F>
class DecomposableT
  : public Decomposable,
    public ModelEntropy<typename F::real_type> {

public:
  using real_type = typename F::real_type;

  DecomposableT() {
    vt_ = F::vtable.copy<FactorTraits>();
  }

  const F& potential(Vertex* v) const {
    return Decomposable::potential(v).cast<F>();
  }

  const F& potential(Edge* e) const {
    return Decomposable::potential(e).cast<F>();
  }

  F marginal(const Domain& domain) const {
    return Decomposable::marginal(domain).cast<F>();
  }

  void reset(const FactorGraph<F>& factors) {
    Decomposable::reset(factors);
  }

  void reset(const std::vector<Domain>& domains, const std::vector<F>& factors) {
    Decomposable::reset(domains, factor.data(), factors.data() + factors.size());
  }

  void reset(const Domain& domain, const F& factor) const {
    Decomposable::reset(domain, factor.cast<F>());
  }

  void multiply_in(const FactorGraph<F>& factors) {
    Decomposable::multiply_in(factors);
  }

  void multiply_in(const std::vector<Domain>& domains, const std::vector<F>& factors) {
    Decomposable::multiply_in(domains, factors.data(), factors.data() + factors.size());
  }

  void multiply_in(const Domain& domain, const F& factor) {
    Decomposable::multiply_in(domain, factor.cast<Potential>());
  }

  /// Returns the entropy of the entire model.
  real_type entropy() const {
    real_type result(0);
    for (Vertex* v : vertices()) {
      result += potential(v).entropy();
    }
    for (Edge* e : edges()) {
      result -= potential(e).entropy();
    }
    return result;
  }

  /// Compute the entropy for a subset of arguments.
  real_type entropy(const Domain& domain) const override {
    // First try to compute the entropy directly from the edge / node marginal
    Edge* e = find_separator_cover(domain);
    if (e) {
      return potential(e).entropy(dims(e, domain));
    }

    Vertex* v = find_cluster_cover(domain);
    if (v) {
      return potential(e).entropy(dims(v, domain));
    }

    // Failing that, compute the marginal of the model
    Decomposable<F> tmp;
    marginal(domain, tmp);
    return tmp.entropy();
  }

  /// Computes the log-likelihood of an assignment.
  real_type Decomposable::log(const Assignment& a) const {
    real_type result(0);
    for (Vertex* v : vertices()) {
      result += potential(v).log(a.values(cluster(v)));
    }
    for (edge_descriptor e : edges()) {
      result -= potential(e).log(a.values(separator(e)));
    }
    return result;
  }
}; // class DecomposableT

} // namespace libgm
