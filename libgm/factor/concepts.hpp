#ifndef LIBGM_FACTOR_CONCEPTS_HPP
#define LIBGM_FACTOR_CONCEPTS_HPP

#include <iostream>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <libgm/copy_ptr.hpp>
#include <libgm/global.hpp>
#include <libgm/learning/validation/crossval_parameters.hpp>
#include <libgm/range/concepts.hpp>
#include <libgm/stl_concepts.hpp>

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
    typedef typename F::index_type index_type;

    /**
     * Returns the value of the factor for the given index.
     * For conditional factors, the index is specified in the order
     * (head, tail).
     */
    typename F::result_type operator()(const index_type& index) const;

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
  

  //============================================================================
  // CONDITIONAL FACTORS
  //============================================================================

  /**
   * The concept that represents a CRF factor/potential.
   *
   * A CRF factor is an arbitrary function Phi(Y,X) which is part of a CRF model
   * P(Y | X) = (1/Z(X)) \prod_i Phi_i(Y_{C_i},X_{C_i}).
   * This allows support of a variety of factors, such as:
   *  - a table_factor over finite variables in X,Y
   *  - a logistic regression function which supports vector variables in X
   *  - Gaussian factors
   *
   * CRF factors can be arbitrary functions, but it is generally easier to think
   * of a CRF factor as an exponentiated sum of weights times feature values:
   *  Phi(Y,X) = \exp[ \sum_j w_j * f_j(Y,X) ]
   * where w_j are fixed (or learned) weights and f_j are arbitrary functions.
   * Since CRF parameter learning often requires the parameters to be in
   * log-space but inference often requires the parameters to be in real-space,
   * CRF factors support both:
   *  - They maintain a bit indicating whether their data is stored
   *    in log-space.
   *  - They have a method which tries to change the internal format
   *    between log- and real-space.
   *  - The learning methods explicitly state what space they return values in.
   * These concepts from parameter learning are in CRFfactor since it makes
   * things more convenient for crf_model (instead of keeping the learning
   * concepts within the LearnableCRFfactor concept class).
   *
   * @author Joseph Bradley
   *
   * @todo Clean this up.
   */
  template <class F>
  struct CRFfactor
    : DefaultConstructible<F>, CopyConstructible<F>, Assignable<F> {

    // Public types
    // =========================================================================

    //! The type that represents the value returned by factor's
    //! operator() and norm_constant().  Typically, this type is either
    //! double or logarithmic<double>.
    typedef typename F::result_type result_type;

    //! The type of input variables used by the factor.
    //! Typically, this type is either libgm::variable or its descendant.
    typedef typename F::input_variable_type input_variable_type;

    //! The type of output variables used by the factor.
    //! Typically, this type is either libgm::variable or its descendant.
    typedef typename F::output_variable_type output_variable_type;

    //! The type of variables used by the factor.
    //! Typically, this type is either libgm::variable or its descendant.
    //! Both input_variable_type and output_variable_type should inherit
    //! from this type.
    typedef typename F::variable_type variable_type;

    //! The type that represents the factor's input variable domain,
    //! that is, the set of input arguments X in the factor.
    //! This type must be equal to set<input_variable_type>.
    typedef typename F::input_domain_type input_domain_type;

    //! The type that represents the factor's output variable domain,
    //! that is, the set of output arguments Y in the factor.
    //! This type must be equal to set<output_variable_type>.
    typedef typename F::output_domain_type output_domain_type;

    //! The type that represents the factor's variable domain,
    //! that is, the set of arguments X,Y in the factor.
    //! This type must be equal to set<variable_type>.
    typedef typename F::domain_type domain_type;

    //! The type that represents an assignment to input variables.
    typedef typename F::input_assignment_type input_assignment_type;

    //! The type that represents an assignment to output variables.
    typedef typename F::output_assignment_type output_assignment_type;

    //! The type that represents an assignment.
    typedef typename F::assignment_type assignment_type;

    //! The type which this factor f(Y,X) outputs to represent f(Y, X=x).
    //! For finite Y, this will probably be table_factor;
    //! for vector Y, this will probably be a subtype of gaussian_base.
    typedef typename F::output_factor_type output_factor_type;

    //! Type which parametrizes this factor, usable for optimization and
    //! learning.
    typedef typename F::optimization_vector optimization_vector;

    // Public methods: Constructors, getters, helpers
    // =========================================================================

    //! @return  output variables in Y for this factor
    const output_domain_type& output_arguments() const;

    //! @return  input variables in X for this factor
    const input_domain_type& input_arguments() const;

    //! @return  input variables in X for this factor
    copy_ptr<input_domain_type> input_arguments_ptr() const;

    //! It may be faster to use input_arguments(), output_arguments().
    //! @return  variables in Y,X for this factor
    domain_type arguments() const;

    //! @return  true iff the data is stored in log-space
    bool log_space() const;

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space();

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space();

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    const optimization_vector& weights() const;

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    optimization_vector& weights();

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! (default after construction = false)
    bool fixed_value() const;

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! This returns a mutable reference.
    //! (default after construction = false)
    bool& fixed_value();

    void print(std::ostream& out) const;

    // Public methods: Probabilistic queries
    // =========================================================================

    //! If this factor is f(Y,X), compute f(Y, X = x).
    //!
    //! @param a  This must assign values to all X in this factor
    //!           (but may assign values to any other variables as well).
    //! @return  table factor representing the factor with
    //!          the given input variable (X) instantiation
    const output_factor_type& condition(const assignment& a) const;

    //! If this factor is f(Y,X), compute f(Y, X = x).
    //!
    //! @param r Record with values for X in this factor
    //!          (which may have values for any other variables as well).
    //! @return  table factor representing the factor with
    //!          the given input variable (X) instantiation
//    const output_factor_type& condition(const record& r) const;

    //! Returns the empirical expectation of the log of this factor.
    //! In particular, if this factor represents P(A|B), then
    //! this returns the expected log likelihood of the distribution P(A | B).
//    double log_expected_value(const dataset& ds) const;

    concept_usage(CRFfactor) {
      libgm::same_type(f_const_ref.output_arguments(), od_const_ref);
      libgm::same_type(f_const_ref.input_arguments(), id_const_ref);
      libgm::same_type(f_const_ref.input_arguments_ptr(), id_copy_ptr);
      libgm::same_type(f_const_ref.arguments(), dom);
      libgm::same_type(f_const_ref.log_space(), b);
      libgm::same_type(f_ptr->convert_to_log_space(), b);
      libgm::same_type(f_ptr->convert_to_real_space(), b);
      libgm::same_type(f_const_ref.weights(), opt_vec_const_ref);
      libgm::same_type(f_ptr->weights(), opt_vec_ref);
      libgm::same_type(f_const_ref.fixed_value(), b);
      libgm::same_type(f_ref.fixed_value(), b_ref);
      f_const_ref.print(out);

      libgm::same_type(f_const_ref.condition(a_const_ref), of_const_ref);
//      libgm::same_type(f_const_ref.condition(rec_const_ref), of_const_ref);
//      libgm::same_type(f_const_ref.log_expected_value(ds_const_ref), d);
      out << f_const_ref;
    }

  private:
    static F& f_ref;
    static const F& f_const_ref;
    static const output_domain_type& od_const_ref;
    static const input_domain_type& id_const_ref;
    copy_ptr<input_domain_type> id_copy_ptr;
    domain_type dom;
    bool b;
    static bool& b_ref;
    F* f_ptr;
    static const optimization_vector& opt_vec_const_ref;
    static optimization_vector& opt_vec_ref;
    static std::ostream& out;

//    static const record& rec_const_ref;
    static const output_factor_type& of_const_ref;

    static const assignment& a_const_ref;
//    static const dataset& ds_const_ref;
    double d;

  };  // struct CRFfactor

  /**
   * CRFfactor which supports gradient-based parameter learning,
   * as well as structure learning using pwl_crf_learner.
   *
   * @todo Once factors are templatized by linear algebra type,
   *       add back methods here which use datasets/records.
   */
  /*
  template <class F>
  struct LearnableCRFfactor
    : public CRFfactor<F> {

    // Public types
    // =========================================================================

    // Import types from base class
    typedef typename CRFfactor<F>::result_type result_type;
    typedef typename CRFfactor<F>::input_variable_type input_variable_type;
    typedef typename CRFfactor<F>::output_variable_type output_variable_type;
    typedef typename CRFfactor<F>::variable_type variable_type;
    typedef typename CRFfactor<F>::input_domain_type input_domain_type;
    typedef typename CRFfactor<F>::output_domain_type output_domain_type;
    typedef typename CRFfactor<F>::domain_type domain_type;
    typedef typename CRFfactor<F>::input_assignment_type input_assignment_type;
    typedef typename CRFfactor<F>::output_assignment_type
      output_assignment_type;
    typedef typename CRFfactor<F>::assignment_type assignment_type;
    typedef typename CRFfactor<F>::output_factor_type output_factor_type;
    typedef typename CRFfactor<F>::optimization_vector optimization_vector;

    //! Type of parameters passed to learning methods.
    //! This should have at least this value:
    //!  - regularization_type reg
    typedef typename F::parameters parameters;

    //! Regularization information.  This should contain, e.g., the type
    //! of regularization being used and the strength.
    //! This should have 3 values:
    //!  - size_t regularization: type of regularization
    //!  - static const size_t nlambdas: dimensionality of lambdas
    //!  - vec lambdas: regularization parameters
    typedef typename F::regularization_type regularization_type;

    // Public methods: Learning methods
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void
    add_gradient(optimization_vector& grad, const record& r, double w) const;

    //! Adds the expectation of the gradient of the log of this factor
    //! w.r.t. the weights, evaluated with the current weights and at the
    //! given datapoint for the X values.  The expectation is over the Y
    //! values and w.r.t. the given factor's distribution.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param fy     Distribution over (at least) the Y variables in this
    //!               factor.
    //! @param w      Weight by which to multiply the added values.
    //! @tparam YFactor  Factor type for a distribution over Y variables.
    void add_expected_gradient(optimization_vector& grad, const record& r,
                               const output_factor_type& fy, double w) const;

    //! This is equivalent to (but faster than) calling:
    //!   add_gradient(grad, r, w);
    //!   add_expected_gradient(grad, r, fy, -1 * w);
    void add_combined_gradient(optimization_vector& grad, const record& r,
                               const output_factor_type& fy, double w) const;

    //! Adds the diagonal of the Hessian of the log of this factor w.r.t. the
    //! weights, evaluated at the given datapoint with the current weights.
    //! @param hessian Pre-allocated vector to which to add the hessian.
    //! @param r       Datapoint.
    //! @param w       Weight by which to multiply the added values.
    void
    add_hessian_diag(optimization_vector& hessian, const record& r,
                     double w) const;

    //! Adds the expectation of the diagonal of the Hessian of the log of this
    //! factor w.r.t. the weights, evaluated with the current weights and at the
    //! given datapoint for the X values.  The expectation is over the Y
    //! values and w.r.t. the given factor's distribution.
    //! @param hessian Pre-allocated vector to which to add the Hessian.
    //! @param r       Datapoint.
    //! @param fy      Distribution over (at least) the Y variables in this
    //!                factor.
    //! @param w       Weight by which to multiply the added values.
    template <typename YFactor>
    void
    add_expected_hessian_diag(optimization_vector& hessian, const record& r,
                              const YFactor& fy, double w) const;

    //! Adds the expectation of the element-wise square of the gradient of the
    //! log of this factor w.r.t. the weights, evaluated with the current
    //! weights and at the given datapoint for the X values. The expectation is
    //! over the Y values and w.r.t. the given factor's distribution.
    //! @param sqrgrad Pre-allocated vector to which to add the squared gradient.
    //! @param r       Datapoint.
    //! @param fy      Distribution over (at least) the Y variables in this
    //!                factor.
    //! @param w       Weight by which to multiply the added values.
    template <typename YFactor>
    void
    add_expected_squared_gradient(optimization_vector& sqrgrad, const record& r,
                                  const YFactor& fy, double w) const;

    //! Returns the regularization penalty for the current weights and
    //! the given regularization parameters.
    double regularization_penalty(const regularization_type& reg) const;

    //! Adds the gradient of the regularization term for the current weights
    //! and the given regularization parameters.
    //! @param w       Weight by which to multiply the added values.
    void add_regularization_gradient(optimization_vector& grad,
                                     const regularization_type& reg,
                                     double w) const;

    //! Adds the diagonal of the Hessian of the regularization term for the
    //! current weights and the given regularization parameters.
    //! @param w       Weight by which to multiply the added values.
    void add_regularization_hessian_diag(optimization_vector& hd,
                                         const regularization_type& reg,
                                         double w) const;

    concept_usage(LearnableCRFfactor) {
      libgm::same_type(params_const_ref.valid(), b);
      libgm::same_type(params_const_ref.reg, reg);

      libgm::same_type(reg.regularization, tmpsize);
      libgm::same_type(reg.nlambdas, tmpsize);
      libgm::same_type(reg.lambdas, tmpvec);

      f_const_ref.add_gradient(opt_vec_ref, rec_const_ref, d);
      f_const_ref.add_expected_gradient(opt_vec_ref, rec_const_ref,
                                        of_const_ref, d);
//      f_const_ref.add_combined_gradient(opt_vec_ref, rec_const_ref,
//                                        of_const_ref, d);
      f_const_ref.add_hessian_diag(opt_vec_ref, rec_const_ref, d);
      f_const_ref.add_expected_hessian_diag(opt_vec_ref, rec_const_ref,
                                            of_const_ref, d);
      f_const_ref.add_expected_squared_gradient(opt_vec_ref, rec_const_ref,
                                                of_const_ref, d);

      libgm::same_type(d, f_const_ref.regularization_penalty(reg));
      f_const_ref.add_regularization_gradient(opt_vec_ref, reg, d);
      f_const_ref.add_regularization_hessian_diag(opt_vec_ref, reg, d);
    }

  private:
    static const F& f_const_ref;
    static const output_domain_type& od_const_ref;
    static const input_domain_type& id_const_ref;
    copy_ptr<input_domain_type> id_copy_ptr;
    domain_type dom;
    bool b;
    F* f_ptr;
    static const optimization_vector& opt_vec_const_ref;
    static optimization_vector& opt_vec_ref;
    static std::ostream& out;
    static const record& rec_const_ref;
    static const output_factor_type& of_const_ref;
    static const assignment& a_const_ref;
//    static const dataset& ds_const_ref;
    double d;

    static const parameters& params_const_ref;

    size_t tmpsize;
    vec tmpvec;

    static const regularization_type& reg;
//    boost::shared_ptr<dataset> ds_shared_ptr;
    unsigned uns;

    output_factor_type of;

    static std::vector<regularization_type>& reg_params;
    static vec& vec_ref;
    static const vec& vec_const_ref;

    static const crossval_parameters& cv_params_const_ref;

  }; // struct LearnableCRFfactor
  */

} // namespace libgm

#endif // #if 0

#endif

