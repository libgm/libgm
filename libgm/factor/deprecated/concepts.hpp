#ifndef LIBGM_FACTOR_CONCEPTS_HPP
#define LIBGM_FACTOR_CONCEPTS_HPP

#if 0

namespace libgm {

  //============================================================================
  // MARGINAL FACTORS
  //============================================================================

  /**
   * A concept that represents a factor.
   *
   * A factor needs to be default constructible, copy constructible,
   * assignable, and implement a unary function from its domain to the
   * range. A factor needs to provide a number of elementary operations:
   * a non-modifying binary combine operation, unary restrict and collapse
   * operations, as well as modifying normalize and variable substitution
   * operations.
   *
   * The library defines a number of convenience free (non-member)
   * functions that call the basic factor operations described above.
   * These functions are automatically included in the factor.hpp header.
   *
   * Typically, factors define constructors and operators that convert
   * from / to other types. For example, a table_factor defines a
   * conversion constructor from a constant_factor.  It also defines a
   * conversion operator to constant_factor (this latter conversion
   * fails if the table_factor has non-empty argument set). By
   * convention, the more complex factors define the conversions
   * from/to the more primitive factors. Since the conversion from a
   * primitive datatype is always valid, it is not declared as
   * explicit. The conversion to a primitive datatype may not be
   * valid, as indicated above in the case of converting a
   * table_factor to a constant_factor.  Such conversions are always
   * checked at runtime.
   *
   * \ingroup factor_concepts
   * \see constant_factor, table_factor
   *
   * \todo Swappable
   */
  template <typename F>
  struct Factor
    : DefaultConstructible<F>, CopyConstructible<F>, Assignable<F> {

    /**
     * The type that represents the value returned by factor's
     * operator() and norm_constant(). Typically, this type is either
     * double or logarithmic<double>.
     */
    typedef typename F::result_type result_type;

    /**
     * The type that represents a real number for the result of
     * calling log(operator()). This should be either double or float.
     */
    typedef typename F::real_type real_type;

    /**
     * The type of variables, used by the factor. Typically, this type is
     * either libgm::variable or its descendant.
     */
    typedef typename F::variable_type variable_type;

    /**
     * The type that represents the factor's domain, that is, the set of
     * factor's arguments. This type must be equal to set<variable_type>.
     */
    typedef typename F::domain_type domain_type;

    /**
     * The type that represents the factor's argument vector, that is,
     * a sequence of variables. This must be equal to vector<variable_type>.
     */
    typedef typename F::var_vector_type var_vector_type;

    /**
     * The type that represents an assignment to a subset of variables.
     * This type can be used to evaluate the factor via operator().
     */
    typedef typename F::assignment_type assignment_type;

    /**
     * Default constructor for a factor with no arguments and factor-specific value.
     */
    Factor();

    /**
     * Conversion from a constant: creates a factor with no arguments
     * and equal to the given constant.
     */
    explicit Factor(result_type value);

    /**
     * Creates a factor with the given argument set and factor-specific value.
     */
    explicit Factor(const domain_type& args);

    /**
     * Creates a factor with the given argument set and given likelihood.
     */
    Factor(const domain_type& args, result_type likelihood);

    /**
     * Returns the arguments of the factor.
     */
    const domain_type& arguments() const;

    /**
     * Returns the value of the factor for the given assignment.
     */
    result_type operator()(const assignment_type& a) const;

    /**
     * Returns a new factor which represents the result of restricting this
     * factor to an assignment to some of its arguments.
     * \todo standardize the other versions
     */
    F restrict(const assignment_type& a) const;

    /**
     * Renames the arguments of this factor in-place.
     *
     * \param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     * \todo Requires that the keys and values of var_map are disjoint?
     */
    F& subst_args(const std::map<variable_type, variable_type>& map);

    // TODO: fix this
    concept_usage(Factor) {
      F f;
      const F& cf = f;
      std::map<variable_type, variable_type> vm;
      assignment_type a;
      domain_type d;

      // member functions
      libgm::same_type(cf.arguments(), d);
      libgm::same_type(f.subst_args(vm), f);

      // static functions
      f.restrict(a);
    }

  }; // concept Factor

  /**
   * A concept that preresents a factor that can be indexed efficiently.
   * The factor defines a factor-specific index type that can be used to
   * evaluate the factor using operator(). For conditional factors, the
   * ordering of variables in the index is always (head, tail).
   *
   * Even if the factor is not IndexableFactor, its value an be still evaluated
   * by passing assignment_type to operator(), though this operation may not
   * be efficient.
   */
  template <typename F>
  struct IndexableFactor : Factor<F> {
    /**
     * The type that represents an assignment to all of factors' arguments
     * in their natural order. This type can be often used to evaluate the
     * factor efficiently.
     */
    typedef typename F::vector_type vector_type;

    /**
     * Returns the value of the factor for the given index.
     * For conditional factors, the index is specified in the order
     * (head, tail).
     */
    typename F::result_type operator()(const vector_type& index) const;

    /**
     * Returns a factor that is equivalent to this factor, but has the
     * variables ordered according to vars.
     * \deprecated This will no longer be necessary / desirable once we
     *             unify Factor and CRFfactor.
     */
    F reorder(const typename F::var_vector_type& vars) const;

  };

  /**
   * A concept that represents a factor of a probability distribution.
   * This factor may represent a (possibly unnormalized) marginal
   * probability distribution alpha * p(A), or a (possibly unnormalized)
   * conditional probability distribution alpha * P(A|B).
   *
   * \ingroup factor_concepts
   * \see table_factor
   */
  template <typename F>
  struct DistributionFactor : Factor<F> {

    /**
     * A functor that, given a set of arguments, returns a marginal
     * distribution over these arguments. Must be identical to
     * boost::function<F(const domain_type&)>.
     */
    typedef typename F::marginal_fn_type marginal_fn_type;

    /**
     * A functor that, given a set of head and tail arguments, returns
     * the conditional distribution p(head|tail). Must be identical to
     * boost::function<F(const domain_type&, const domain_type&)>.
     */
    typedef typename F::conditional_fn_type conditional_fn_type;

    /**
     * Multiplies this factor by another factor of the same kind.
     */
    F& operator*=(const F& f);

    /**
     * Multiplies this factor by a constant. This changes the total probability
     * of the distribution represented by this factor.
     */
    F& operator*=(typename F::result_type value);

    /**
     * Divides this factor by a constant. This changes the total probability
     * of the distribution represented by this factor.
     */
    F& operator/=(typename F::result_type value);

    /**
     * Computes a marginal (sum) over a subset of variables
     * \param retain A set of arguments that should be retained. This may
     *        include arguments not present in the factor (which are ignored).
     * \todo standardize the other versions
     */
    F marginal(const typename F::domain_type& retain) const;

    /**
     * If this factor represents the marginal distribution P(A,B),
     * then this returns P(A|B).
     */
    F conditional(const typename F::domain_type& tail) const;

    /**
     * Returns true if this factor represents a conditional distribution
     * p(args - tail | tail). This function may be implemented numerically
     * and may take a second optional argument, which is the tolerance.
     */
    bool is_conditional(const typename F::domain_type& tail) const;

    /**
     * Returns true if the factor is normalizable (that is, it has a positive
     * and finite mass.
     */
    bool is_normalizable() const;

    /**
     * Returns the normalization constant of this factor. Dividing the factor
     * by this constant would make it a proper distribution.
     */
    typename F::result_type norm_constant() const;

    /**
     * Normalizes a factor in-place. This function throws an exception if
     * the supplied factor is not normalizable (because its integral is
     * not positive and finite).
     */
    F& normalize();

    /**
     * Computes the entropy of the distribution, provided that this factor
     * represents a marginal distribution.
     * \todo standardize the other versions
     */
    typename F::real_type entropy() const;

    /**
     * Computes the Kullback-Liebler divergence from this factor to the
     * supplied factor. Assumes that both *this and q represent a marginal
     * distribution.
     * \todo standardize the other versions
     */
    typename F::real_type relative_entropy(const F& q);

    /**
     * Computes the mutual information between two sets of variables
     * in this factor's arguments.
     */
    typename F::real_type mutual_information(const typename F::domain_type&,
                                             const typename F::domain_type&) const;

    /**
     * Draws random samples from the distribution represented by this factor.
     * Requires that the factor represents a marginal distribution.
     */
    template <typename RandomNumberGenerator>
    typename F::assignment_type sample(RandomNumberGenerator& rng) const;

    /**
     * A free function that multipies two distribution factors.
     */
    friend F operator*(const F& f, const F& g);

    /**
     * A free function that multiplies a distribution factor and a constant.
     */
    friend F operator*(const F& f, typename F::result_type x);

    /**
     * A free function that multiplies a distributino factor and a constant.
     */
    friend F operator*(typename F::result_type x, const F& f);

    // TODO: fix this
    concept_usage(DistributionFactor) {
      bool b;
      typename F::result_type r;
      F f;
      const F& cf = f;

      // factor operations
      libgm::same_type(f, f*=f);
      libgm::same_type(f, f.marginal(typename F::domain_type()));
      libgm::same_type(b, f.is_normalizable());
      libgm::same_type(f.normalize(), f);
      r = f.norm_constant();
      libgm::same_type(r, r); // avoid unused-but-set-variable warning

      // entropy computations
      cf.entropy();
      cf.relative_entropy(cf);
    }

  }; // concept DistributionFactor

  /**
   * A class that represents a marginal or conditiona distribution that can be
   * learned from data. Each LearnarbleFactor is associated with a dataset type
   * (which is usually an abstract base class) which represents a collection of
   * data points in a format accessible to this factor.
   */
  template <typename F>
  class LearnableFactor : public DistributionFactor<F> {

    /**
     * The type that represents a dataset compatible with this factor.
     * Must satisfy the Dataset concept.
     */
    typedef typename F::dataset_type dataset_type;

    /**
     * The type that represents a record in a dataset compatible with
     * this factor.
     * \deprecated This is part of the old dataset framework and is deprecated.
     */
    typedef typename F::record_type record_type;

    /**
     * Returns the value of the factor for the given dataset record.
     * \deprecated This is part of the old dataset framework and is deprecated.
     */
    typename F::result_type operator()(const record_type& record) const;

  };

} // namespace libgm

#endif // #if 0

#endif

