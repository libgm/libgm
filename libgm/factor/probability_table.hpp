#ifndef LIBGM_PROBABILITY_TABLE_HPP
#define LIBGM_PROBABILITY_TABLE_HPP

#include <libgm/factor/base/table_factor.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/likelihood/probability_table_ll.hpp>
#include <libgm/math/likelihood/probability_table_mle.hpp>
#include <libgm/math/random/table_distribution.hpp>

#include <initializer_list>
#include <iostream>
#include <random>

namespace libgm {

  // Forward declaration
  template <typename Arg, typename T> class canonical_table;
  template <typename Arg, std::size_t N, typename T> class probability_array;
  template <typename Arg, std::size_t N, typename T> class canonical_array;

  /**
   * A factor of a categorical probability distribution in the probability
   * space. This factor represents a non-negative function over finite
   * variables X directly using its parameters, f(X = x | \theta) = \theta_x.
   * In some cases, e.g. in a Bayesian network, this factor in fact
   * represents a (conditional) probability distribution. In other cases,
   * e.g. in a Markov network, there are no constraints on the normalization
   * of f.
   *
   * \tparam T a real type representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename T = double>
  class probability_table : public table_factor<Arg, T> {
  public:
    // Public types
    //==========================================================================

    // Factor member types
    typedef T                    real_type;
    typedef T                    result_type;
    typedef Arg                  argument_type;
    typedef domain<Arg>          domain_type;
    typedef uint_assignment<Arg> assignment_type;

    // ParametricFactor types
    typedef table<T>    param_type;
    typedef uint_vector vector_type;
    typedef table_distribution<T> distribution_type;

    // LearnableDistributionFactor types
    typedef probability_table_ll<T>  ll_type;
    typedef probability_table_mle<T> mle_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    probability_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit probability_table(const domain_type& args) {
      this->reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit probability_table(T value) {
      this->reset();
      this->param_[0] = value;
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_table(const domain_type& args, T value) {
      this->reset(args);
      this->param_.fill(value);
    }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain_type& args, const table<T>& param)
      : table_factor<Arg, T>(args, param) { }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain_type& args, table<T>&& param)
      : table_factor<Arg, T>(args, std::move(param)) { }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const domain_type& args,
                      std::initializer_list<T> values) {
      this->reset(args);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), this->begin());
    }

    //! Conversion from a canonical_table factor.
    explicit probability_table(const canonical_table<Arg, T>& f) {
      *this = f;
    }

    //! Conversion from a probability_array factor.
    template <std::size_t N>
    explicit probability_table(const probability_array<Arg, N, T>& f) {
      this->reset(f.arguments());
      std::copy(f.begin(), f.end(), this->begin());
    }

    //! Conversion from a canonical_array factor.
    template <std::size_t N>
    explicit probability_table(const canonical_array<Arg, N, T>& f) {
      this->reset(f.arguments());
      std::transform(f.begin(), f.end(), this->begin(), exponent<T>());
    }

    //! Assigns a constant to this factor.
    probability_table& operator=(T value) {
      this->reset();
      this->param_[0] = value;
      return *this;
    }

    //! Assigns a probability table factor to this factor.
    probability_table& operator=(const canonical_table<Arg, T>& f) {
      this->reset(f.arguments());
      std::transform(f.begin(), f.end(), this->begin(), exponent<T>());
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(probability_table& f, probability_table& g) {
      f.base_swap(g);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return this->finite_args_;
    }

    //! Returns the value of the factor for the given assignment.
    T operator()(const assignment_type& a) const {
      return this->param(a);
    }

    //! Returns the value of the factor for the given index.
    T operator()(const uint_vector& index) const {
      return this->param(index);
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const assignment_type& a) const {
      return std::log(this->param(a));
    }

    //! Returns the log-value of the factor for the given index.
    T log(const uint_vector& index) const {
      return std::log(this->param(index));
    }

    //! Returns true if the two factors have the same arguments and values.
    friend bool
    operator==(const probability_table& f, const probability_table& g) {
      return f.arguments() == g.arguments() && f.param() == g.param();
    }

    //! Returns true if the factors do not have the same arguments or values.
    friend bool
    operator!=(const probability_table& f, const probability_table& g) {
      return !(f == g);
    }

    // Factor operations
    //==========================================================================

    //! Element-wise addition of two factors.
    probability_table& operator+=(const probability_table& f) {
      this->transform_inplace(f, std::plus<T>());
      return *this;
    }

    //! Element-wise subtraction of two factors.
    probability_table& operator-=(const probability_table& f) {
      this->transform_inplace(f, std::minus<T>());
      return *this;
    }

    //! Multiplies another factor into this one.
    probability_table& operator*=(const probability_table& f) {
      this->join_inplace(f, std::multiplies<T>());
      return *this;
    }

    //! Divides another factor into this one.
    probability_table& operator/=(const probability_table& f) {
      this->join_inplace(f, safe_divides<T>());
      return *this;
    }

    //! Increments this factor by a constant.
    probability_table& operator+=(T x) {
      this->param_.transform(incremented_by<T>(x));
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_table& operator-=(T x) {
      this->param_.transform(decremented_by<T>(x));
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_table& operator*=(T x) {
      this->param_.transform(multiplied_by<T>(x));
      return *this;
    }

    //! Divides this factor by a constant.
    probability_table& operator/=(T x) {
      this->param_.transform(divided_by<T>(x));
      return *this;
    }

    //! Element-wise sum of two factors.
    friend probability_table
    operator+(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, std::plus<T>());
    }

    //! Element-wise difference of two factors.
    friend probability_table
    operator-(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, std::minus<T>());
    }

    //! Multiplies two probability_table factors.
    friend probability_table
    operator*(const probability_table& f, const probability_table& g) {
      return join<probability_table>(f, g, std::multiplies<T>());
    }

    //! Divides two probability_table factors.
    friend probability_table
    operator/(const probability_table& f, const probability_table& g) {
      return join<probability_table>(f, g, safe_divides<T>());
    }

    //! Adds a probability_table factor and a constant.
    friend probability_table
    operator+(const probability_table& f, T x) {
      return transform<probability_table>(f, incremented_by<T>(x));
    }

    //! Adds a probability_table factor and a constant.
    friend probability_table
    operator+(T x, const probability_table& f) {
      return transform<probability_table>(f, incremented_by<T>(x));
    }

    //! Subtracts a constant from a probability_table factor.
    friend probability_table
    operator-(const probability_table& f, T x) {
      return transform<probability_table>(f, decremented_by<T>(x));
    }

    //! Subtracts a probability_table factor from a constant.
    friend probability_table
    operator-(T x, const probability_table& f) {
      return transform<probability_table>(f, subtracted_from<T>(x));
    }

    //! Multiplies a probability_table factor by a constant.
    friend probability_table
    operator*(const probability_table& f, T x) {
      return transform<probability_table>(f, multiplied_by<T>(x));
    }

    //! Multiplies a probability_table factor by a constant.
    friend probability_table
    operator*(T x, const probability_table& f) {
      return transform<probability_table>(f, multiplied_by<T>(x));
    }

    //! Divides a probability_table factor by a constant.
    friend probability_table
    operator/(const probability_table& f, T x) {
      return transform<probability_table>(f, divided_by<T>(x));
    }

    //! Divides a constant by a probability_table factor.
    friend probability_table
    operator/(T x, const probability_table& f) {
      return transform<probability_table>(f, dividing<T>(x));
    }

    //! Raises the probability_table factor by an exponent.
    friend probability_table
    pow(const probability_table& f, T x) {
      return transform<probability_table>(f, power<T>(x));
    }

    //! Element-wise maximum of two factors.
    friend probability_table
    max(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, libgm::maximum<T>());
    }

    //! Element-wise minimum of two factors.
    friend probability_table
    min(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, libgm::minimum<T>());
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend probability_table
    weighted_update(const probability_table& f,
                    const probability_table& g, T a) {
      return transform<probability_table>(f, g, weighted_plus<T>(1 - a, a));
    }

    //! Computes the marginal of the factor over a subset of variables.
    probability_table marginal(const domain_type& retain) const {
      probability_table result;
      marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    probability_table maximum(const domain_type& retain) const {
      probability_table result;
      maximum(retain, result);
      return result;
    }

    //! Computes the minimum for each assignment to the given variables.
    probability_table minimum(const domain_type& retain) const {
      probability_table result;
      minimum(retain, result);
      return result;
    }

    //! If this factor represents p(x, y), returns p(x | y).
    //! \todo reorder the variables, so that tail comes last
    probability_table conditional(const domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const domain_type& retain, probability_table& result) const {
      this->aggregate(retain, T(0), std::plus<T>(), result);
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const domain_type& retain, probability_table& result) const {
      this->aggregate(retain, -inf<T>(), libgm::maximum<T>(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const domain_type& retain, probability_table& result) const {
      this->aggregate(retain, +inf<T>(), libgm::minimum<T>(), result);
    }

    //! Returns the normalization constant of the factor.
    T marginal() const {
      return this->param_.accumulate(T(0), std::plus<T>());
    }

    //! Returns the maximum value in the factor.
    T maximum() const {
      return this->param_.accumulate(-inf<T>(), libgm::maximum<T>());
    }

    //! Returns the minimum value in the factor.
    T minimum() const {
      return this->param_.accumulate(+inf<T>(), libgm::minimum<T>());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    T maximum(assignment_type& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      a.insert_or_assign(arguments(), this->param_.index(it));
      return *it;
    }

    //! Computes the minimum value and stores the corresponding assignment.
    T minimum(assignment_type& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      a.insert_or_assign(arguments(), this->param_.index(it));
      return *it;
    }

    //! Normalizes the factor in-place.
    probability_table& normalize() {
      this->param_ /= marginal();
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return maximum() > 0;
    }

    //! Restricts this factor to an assignment.
    probability_table restrict(const assignment_type& a) const {
      probability_table result;
      restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const assignment_type& a, probability_table& result) const {
      table_factor<Arg, T>::restrict(a, result);
    }

    // Sampling
    //==========================================================================

    //! Returns the distribution with the parameters of this factor.
    table_distribution<T> distribution() const {
      return table_distribution<T>(this->param_);
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    uint_vector sample(Generator& rng) const {
      return sample(rng, uint_vector());
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    uint_vector sample(Generator& rng, const uint_vector& tail) const {
      return this->param_.sample(identity(), rng, tail);
    }

    /**
     * Draws a random sample from a marginal distribution,
     * storing the result in an assignment.
     */
    template <typename Generator>
    void sample(Generator& rng, assignment_type& a) const {
      a.insert_or_assign(arguments(), sample(rng));
    }

    /**
     * Draws a random sample from a conditional distribution,
     * extracting the tail from and storing the result to an assignment.
     * \param ntail the tail variables (must be a suffix of the domain).
     */
    template <typename Generator>
    void sample(Generator& rng, const domain_type& head,
                assignment_type& a) const {
      assert(arguments().prefix(head));
      a.insert_or_assign(head, sample(rng, a.values(arguments(), head.size())));
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return this->param_.transform_accumulate(T(0),
                                               entropy_op<T>(),
                                               std::plus<T>());
    }

    //! Computes the entropy for a subset of variables via marginalization.
    T entropy(const domain_type& a) const {
      return equivalent(arguments(), a) ? entropy() : marginal(a).entropy();
    }

    //! Computes the mutual information between two subsets of this factor's
    //! arguments.
    T mutual_information(const domain_type& a, const domain_type& b) const {
      return entropy(a) + entropy(b) - entropy(a + b);
    }

    //! Computes the cross entropy from p to q.
    friend T
    cross_entropy(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, entropy_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T
    kl_divergence(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, kld_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T
    js_divergence(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, jsd_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }

    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), libgm::maximum<T>());
    }

  }; // class probability_table

  // Input / output
  //============================================================================

  /**
   * Prints a human-readable representation of the table factor to the stream.
   * \relates probability_table
   */
  template <typename Arg, typename T>
  std::ostream&
  operator<<(std::ostream& out, const probability_table<Arg, T>& f) {
    out << "#PT(" << f.arguments() << ")" << std::endl;
    out << f.param();
    return out;
  }

  // Traits
  //============================================================================

  template <typename Arg, typename T>
  struct has_multiplies<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_multiplies_assign<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_divides<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_divides_assign<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_max<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_min<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_marginal<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_maximum<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_minimum<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_arg_max<probability_table<Arg, T> >
    : public std::true_type { };

  template <typename Arg, typename T>
  struct has_arg_min<probability_table<Arg, T> >
    : public std::true_type { };

} // namespace libgm

#endif
