#ifndef LIBGM_CANONICAL_ARRAY_HPP
#define LIBGM_CANONICAL_ARRAY_HPP

#include <libgm/global.hpp>
#include <libgm/factor/base/array_factor.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/eigen.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/operators.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/likelihood/canonical_array_ll.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/random/array_distribution.hpp>

#include <armadillo>
#include <boost/function.hpp>

#include <iostream>

namespace libgm {

  // Forward declarations
  template <typename T, typename Var> class canonical_table;
  template <typename T, size_t N, typename Var> class probability_array;

  /**
   * A factor of a categorical canonical distribution that contains one or
   * two arguments in its domain, where the number of arguments is fixed at
   * compile-time. This factor represents a non-negative function
   * using canonical (natural) parameters \theta of the exponential family as
   * f(x, y | \theta) = exp(\theta_{x,y}) for binary factors and
   * f(x | \theta) = exp(\theta_x) for unary factors. In some cases, e.g.,
   * when used in a Bayesian network, this factor also represents a probability
   * distribution in the log-space.
   * 
   * \tparam T a real type that represents each parameter
   * \tparam N the number of arguments
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T, size_t N, typename Var>
  class canonical_array : public array_factor<T, N, Var> {
  public:
    // Helper types
    typedef array_factor<T, N, Var> base;
    typedef array_domain<Var, 1> unary_domain_type;

    // Factor member types
    typedef T                      real_type;
    typedef logarithmic<T>         result_type;
    typedef Var                    variable_type;
    typedef array_domain<Var, N>   domain_type;
    typedef finite_assignment<Var> assignment_type;

    // ParametricFactor member types
    typedef typename base::array_type param_type;
    typedef finite_index              index_type;
    typedef array_distribution<T, N> distribution_type;
    
    // LearnableFactor types
    typedef canonical_array_ll<T, N> ll_type;

    // ExponentialFamilyFactor member types
    typedef probability_array<T, N, Var> probability_type;
    
    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor. Creates an empty factor.
    canonical_array() { }

    //! Constructs a factor with the given arguments and uninitialized parameters.
    explicit canonical_array(const domain_type& args) {
      this->reset(args);
    }

    //! Constructs a factor with the given arguments and constant value.
    canonical_array(const domain_type& args, logarithmic<T> value) {
      this->reset(args);
      this->param_.fill(value.lv);
    }

    //! Constructs a factor with the given argument and parameters.
    canonical_array(const domain_type& args, const param_type& param)
      : base(args, param) { }

    canonical_array(const domain_type& args, param_type&& param)
      : base(args, std::move(param)) { }

    //! Constructs a factor with the given arguments and parameters.
    canonical_array(const domain_type& args, std::initializer_list<T> values)
      : base(args, values) { }

    //! Conversion from a probability_array factor.
    explicit canonical_array(const probability_array<T, N, Var>& f) {
      *this = f;
    }

    //! Conversion from a canonical_table factor.
    explicit canonical_array(const canonical_table<T, Var>& f) {
      *this = f;
    }

    //! Assigns a probability_array factor to this factor.
    canonical_array& operator=(const probability_array<T, N, Var>& f) {
      this->reset(f.arguments());
      this->param_ = f.param().log();
      return *this;
    }
    
    //! Assigns a canonical_table to this factor
    canonical_array& operator=(const canonical_table<T, Var>& f) {
      this->reset(f.arguments());
      assert(this->size() == f.size());
      std::copy(f.begin(), f.end(), this->begin());
      return *this;
    }

    //! Swaps the content of two canonical_array factors.
    friend void swap(canonical_array& f, canonical_array& g) {
      f.swap(g);
    }

    // Accessors
    //==========================================================================
    //! Returns the value of this factor for an assignment
    logarithmic<T> operator()(const assignment_type& a) const {
      return logarithmic<T>(this->param(a), log_tag());
    }

    //! Returns the value of this factor for an index
    logarithmic<T> operator()(const finite_index& index) const {
      return logarithmic<T>(this->param(index), log_tag());
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const assignment_type& a) const {
      return this->param(a);
    }

    //! Returns the log-value of teh factor for the given index.
    T log(const finite_index& index) const {
      return this->param(index);
    }

    //! Returns true if the two factors have the same argument vectors and values.
    friend bool operator==(const canonical_array& f,
                           const canonical_array& g) {
      return f->equal(g);
    }

    //! Returns true if the two factors do not have the same arguments or values.
    friend bool operator!=(const canonical_array& f,
                           const canonical_array& g) {
      return !f->equal(g);
    }

    // Factor operations
    //==========================================================================

    /**
     * Multiplies another factor with arity M into this one.
     * This operation is only supported when M <= N, i.e.,
     * the given factor has no more arguments than this one.
     */
    template <size_t M>
    typename std::enable_if<M <= N, canonical_array&>::type
    operator*=(const canonical_array<T, M, Var>& f) {
      join_inplace(*this, f, libgm::plus_assign<>());
      return *this;
    }

    /**
     * Divides another factor with arity M into this one.
     * This operation is only supported when M <= N, i.e.,
     * the given factor has no more arguments than this one.
     */
    template <size_t M>
    typename std::enable_if<M <= N, canonical_array&>::type
    operator/=(const canonical_array<T, M, Var>& f) {
      join_inplace(*this, f, libgm::minus_assign<>());
      return *this;
    }

    //! Multiplies this factor by a constant.
    canonical_array& operator*=(logarithmic<T> x) {
      this->param_ += x.lv;
      return *this;
    }

    //! Divides this factor by a constant.
    canonical_array& operator/=(logarithmic<T> x) {
      this->param_ -= x.lv;
      return *this;
    }

    //! Returns the sum of the probabilities of two factors.
    friend canonical_array
    operator+(const canonical_array& f, const canonical_array& g) {
      return transform<canonical_array>(f, g, log_sum_exp<T>());
    }

    //! Multiplies a canonical_array factor by a constant.
    friend canonical_array
    operator*(const canonical_array& f, logarithmic<T> x) {
      return canonical_array(f.arguments(), f.param() + x.lv);
    }

    //! Multiplies a canonical_array factor by a constant.
    friend canonical_array
    operator*(logarithmic<T> x, const canonical_array& f) {
      return canonical_array(f.arguments(), x.lv + f.param());
    }

    //! Divides a canonical_array factor by a constant.
    friend canonical_array
    operator/(const canonical_array& f, logarithmic<T> x) {
      return canonical_array(f.arguments(), f.param() - x.lv);
    }

    //! Divides a constant by a canonical_array factor.
    friend canonical_array
    operator/(logarithmic<T> x, const canonical_array& f) {
      return canonical_array(f.arguments(), x.lv - f.param());
    }

    //! Raises the canonical_array factor by an exponent.
    friend canonical_array
    pow(const canonical_array& f, T x) {
      return canonical_array(f.arguments(), f.param() * x);
    }

    //! Element-wise maximum of two factors.
    friend canonical_array
    max(const canonical_array& f, const canonical_array& g) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(), f.param().max(g.param()));
    }
  
    //! Element-wise minimum of two factors.
    friend canonical_array
    min(const canonical_array& f, const canonical_array& g) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(), f.param().min(g.param()));
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend canonical_array
    weighted_update(const canonical_array& f,
                    const canonical_array& g, T a) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(),
                             (1-a) * f.param() + a * g.param());
    }

    /**
     * Computes the marginal of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array<T, 1, Var> >::type
    marginal(const unary_domain_type& retain) const {
      canonical_array<T, 1, Var> result;
      marginal(retain, result);
      return result;
    }

    /**
     * Computes the maximum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array<T, 1, Var> >::type
    maximum(const unary_domain_type& retain) const { 
      return aggregate<canonical_array<T, 1, Var>>(*this, retain, max_coeff_op());
    }

    /**
     * Computes the minimum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array<T, 1, Var> >::type
    minimum(const unary_domain_type& retain) const {
      return aggregate<canonical_array<T, 1, Var>>(*this, retain, min_coeff_op());
    }

    /**
     * If this factor represents p(x, y) where y = tail, returns p(x | y).
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array>::type
    conditional(const unary_domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    /**
     * Computes the marginal of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    marginal(const unary_domain_type& retain,
             canonical_array<T, 1, Var>& result) const {
      T max = this->param_.maxCoeff();
      transform_aggregate(*this, retain, result,
                          exp_op<T>(-max), log_sum_op<T>(+max));
    }

    /**
     * Computes the maximum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    maximum(const unary_domain_type& retain,
            canonical_array<T, 1, Var>& result) const {
      aggregate(*this, retain, result, max_coeff_op());
    }

    /**
     * Computes the minimum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    minimum(const unary_domain_type& retain,
            canonical_array<T, 1, Var>& result) const {
      aggregate(*this, retain, result, min_coeff_op());
    }

    //! Returns the normalization constant of the factor.
    logarithmic<T> marginal() const {
      T max = this->param_.maxCoeff();
      T sum = std::accumulate(this->begin(), this->end(), T(0),
                              plus_exp<T>(-max));
      return logarithmic<T>(std::log(sum) + max, log_tag());
      // std::log(exp(param()-max).sum())+max is slow (at least on LLVM 3.5)
    }

    //! Returns the maximum value in the factor.
    logarithmic<T> maximum() const {
      return logarithmic<T>(this->param_.maxCoeff(), log_tag());
    }

    //! Returns the minimum value in the factor.
    logarithmic<T> minimum() const {
      return logarithmic<T>(this->param_.minCoeff(), log_tag());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    logarithmic<T> maximum(assignment_type& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Computes the minimum value and stores the corresponding assignment.
    logarithmic<T> minimum(assignment_type& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Normalizes the factor in-place.
    canonical_array& normalize() {
      this->param_ -= marginal().lv;
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return maximum().lv > -inf<T>();
    }
    
    /**
     * Restricts the factor to an assignment and returns the result
     * as a unary factor. This operation is only supported for binary
     * factors, and the assignment must restrict exactly one argument.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array<T, 1, Var> >::type
    restrict(const assignment_type& a) const {
      canonical_array<T, 1, Var> result;
      restrict(a, result);
      return result;
    }

    /**
     * Restricts this factor to an assignment and stores the result
     * in a unary factor. This operation is only supported for binary
     * factors, and teh assignment must restrict exactly one argument.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    restrict(const assignment_type& a,
             canonical_array<T, 1, Var>& result) const {
      restrict_assign(*this, a, result);
    }

    /**
     * Restricts this factor to an assignment, excluding the variables in
     * the unary factor result, and multiplies the restriction into result.
     * This operation must not introduce any new variables and is only
     * supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    restrict_multiply(const assignment_type& a,
                      canonical_array<T, 1, Var>& result) const {
      restrict_join(*this, a, result, plus_assign<>());
    }

    /**
     * Computes the (exponentiated) expected value of the log of this
     * factor w.r.t. the distribution given by a unary probability_array.
     * More precisely, if f(x,y) is this factor and q(y) is the specified
     * factor, returns a factor with parameters E_q[log f(x, Y)].
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, canonical_array<T, 1, Var> >::type
    exp_log(const probability_array<T, 1, Var>& q) const {
      return expectation<canonical_array<T, 1, Var> >(*this, q);
    }

    /**
     * Computes the (exponentiated) expected value of the log of this
     * factor w.r.t. the distribution given by a unary probability_array
     * and multiplies the result to the result factor.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    exp_log_multiply(const probability_array<T, 1, Var>& q,
                     canonical_array<T, 1, Var>& result) const {
      join_expectation(result, *this, q, libgm::plus_assign<>());
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return transform_accumulate(*this, entropy_log_op<T>(), std::plus<T>());
    }

    /**
     * Computes the entropy for a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, T>::type
    entropy(const unary_domain_type& a) const {
      return marginal(a).entropy();
    }

    /**
     * Computes the mutual information between two variables.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, T>::type
    mutual_information(const unary_domain_type& a,
                       const unary_domain_type& b) const {
      assert(subset(a, this->arguments()));
      assert(subset(b, this->arguments()));
      return entropy(a) + entropy(b) - entropy();
    }

    //! Computes the cross entropy from p to q.
    friend T cross_entropy(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, entropy_log_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, kld_log_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, jsd_log_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const canonical_array& p,
                      const canonical_array& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }
    
    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const canonical_array& p,
                      const canonical_array& q) {
      return transform_accumulate(p, q, abs_difference<T>(), libgm::maximum<T>());
    }

  }; // class canonical_array

  /**
   * A canonical_array factor over a single argument using double precision.
   * \relates canonical_array
   */
  typedef canonical_array<double, 1, variable> carray1;

  /**
   * A canonical_array factor over two arguments using double precision.
   * \relates canonical_array
   */
  typedef canonical_array<double, 2, variable> carray2;

  // Input / output
  //============================================================================

  /**
   * Outputs a human-readable representation of the factor to the stream.
   * \relates canonical_array
   */
  template <typename T, size_t N, typename Var>
  std::ostream&
  operator<<(std::ostream& out, const canonical_array<T, N, Var>& f) {
    out << f.arguments() << std::endl
        << f.param() << std::endl;
    return out;
  }

  // Join operation
  //============================================================================

  /**
   * Multiplies two canonical_array factors.
   * \tparam M the arity of the first argument
   * \tparam N the arity of the second argument
   * \return a canonical_array factor whose arity is the maximum of M and N
   * \relates canonical_array
   */
  template <typename T, size_t M, size_t N, typename Var>
  canonical_array<T, (M >= N) ? M : N, Var>
  operator*(const canonical_array<T, M, Var>& f,
            const canonical_array<T, N, Var>& g) {
    typedef canonical_array<T, (M >= N) ? M : N, Var> result_type;
    return join<result_type>(f, g, libgm::plus<>());
  }

  /**
   * Divides two canonical_array factors.
   * \tparam M the arity of the first argument
   * \tparam N the arity of the second argument
   * \return a canonical_array factor whose arity is the maximum of M and N
   * \relates canonical_array
   */
  template <typename T, size_t M, size_t N, typename Var>
  canonical_array<T, (M >= N) ? M : N, Var>
  operator/(const canonical_array<T, M, Var>& f,
            const canonical_array<T, N, Var>& g) {
    typedef canonical_array<T, (M >= N) ? M : N, Var> result_type;
    return join<result_type>(f, g, libgm::minus<>());
  }

} // namespace libgm

#endif
