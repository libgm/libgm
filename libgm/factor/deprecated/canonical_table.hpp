#ifndef LIBGM_CANONICAL_TABLE_HPP
#define LIBGM_CANONICAL_TABLE_HPP

#include <libgm/factor/base/table_factor.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/likelihood/canonical_table_ll.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include <cmath>
#include <initializer_list>
#include <iostream>

namespace libgm {

  // Forward declaration
  template <typename Arg, typename T> class probability_table;

  /**
   * A factor of a categorical probability distribution represented in the
   * canonical form of the exponential family. This factor represents a
   * non-negative function over finite variables X as f(X | \theta) =
   * exp(\sum_x \theta_x * 1(X=x)). In some cases, e.g. in a Bayesian network,
   * this factor also represents a probability distribution in the log-space.
   *
   * \tparam T a real type representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename Arg, typename T = double>
  class canonical_table : public table_factor<Arg, T> {
  public:
    // Public types
    //==========================================================================

    // Factor member types
    typedef T                    real_type;
    typedef logarithmic<T>       result_type;
    typedef Arg                  argument_type;
    typedef domain<Arg>          domain_type;
    typedef uint_assignment<Arg> assignment_type;

    // ParametricFactor types
    typedef table<T>     param_type;
    typedef uint_vector  vector_type;
    typedef multivariate_categorical_distribution<T> distribution_type;

    // LearnableDistributionFactor member types
    typedef canonical_table_ll<T> ll_type;

    // ExponentialFamilyFactor member types
    typedef probability_table<Arg, T> probability_type;

    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    canonical_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit canonical_table(const domain_type& args) {
      this->reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_table(logarithmic<T> value) {
      this->reset();
      this->param_[0] = value.lv;
    }

    //! Constructs a factor with the given arguments and constant value.
    canonical_table(const domain_type& args, logarithmic<T> value) {
      this->reset(args);
      this->param_.fill(value.lv);
    }

    //! Creates a factor with the specified arguments and parameters.
    canonical_table(const domain_type& args, const table<T>& param)
      : table_factor<Arg, T>(args, param) { }

    //! Creates a factor with the specified arguments and parameters.
    canonical_table(const domain_type& args, table<T>&& param)
      : table_factor<Arg, T>(args, std::move(param)) { }

    //! Creates a factor with the specified arguments and parameters.
    canonical_table(const domain_type& args,
                    std::initializer_list<T> values) {
      this->reset(args);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), this->begin());
    }

    //! Conversion from a probability_table factor.
    explicit canonical_table(const probability_table<Arg, T>& f) {
      *this = f;
    }

    //! Assigns a constant to this factor.
    canonical_table& operator=(logarithmic<T> value) {
      this->reset();
      this->param_[0] = value.lv;
      return *this;
    }

    //! Assigns a probability table factor to this factor.
    canonical_table& operator=(const probability_table<Arg, T>& f) {
      this->reset(f.arguments());
      std::transform(f.begin(), f.end(), this->begin(), logarithm<T>());
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(canonical_table& f, canonical_table& g) {
      f.base_swap(g);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments of this factor.
    const domain_type& arguments() const {
      return this->finite_args_;
    }

    //! Returns the value of the factor for the given assignment.
    logarithmic<T> operator()(const assignment_type& a) const {
      return logarithmic<T>(this->param(a), log_tag());
    }

    //! Returns the value of the factor for the given index.
    logarithmic<T> operator()(const uint_vector& index) const {
      return logarithmic<T>(this->param(index), log_tag());
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const assignment_type& a) const {
      return this->param(a);
    }

    //! Returns the log-value of the factor for the given index.
    T log(const uint_vector& index) const {
      return this->param(index);
    }

    //! Returns true if the two factors have the same arguments and values.
    friend bool operator==(const canonical_table& f, const canonical_table& g) {
      return f.arguments() == g.arguments() && f.param() == g.param();
    }

    //! Returns true if the factors do not have the same arguments or values.
    friend bool operator!=(const canonical_table& f, const canonical_table& g) {
      return !(f == g);
    }

    // Factor operations
    //==========================================================================

    //! Multiplies another factor into this one.
    canonical_table& operator*=(const canonical_table& f) {
      this->join_inplace(f, std::plus<T>());
      return *this;
    }

    //! Divides another factor into this one.
    canonical_table& operator/=(const canonical_table& f) {
      this->join_inplace(f, std::minus<T>());
      return *this;
    }

    //! Multiplies this factor by a constant.
    canonical_table& operator*=(logarithmic<T> x) {
      this->param_.transform(incremented_by<T>(x.lv));
      return *this;
    }

    //! Divides this factor by a constant.
    canonical_table& operator/=(logarithmic<T> x) {
      this->param_.transform(decremented_by<T>(x.lv));
      return *this;
    }

    //! Returns the sum of the probabilities of two factors.
    friend canonical_table
    operator+(const canonical_table& f, const canonical_table& g) {
      return transform<canonical_table>(f, g, log_plus_exp<T>());
    }

    //! Multiplies two canonical_table factors.
    friend canonical_table
    operator*(const canonical_table& f, const canonical_table& g) {
      return join<canonical_table>(f, g, std::plus<T>());
    }

    //! Divides two canonical_table factors.
    friend canonical_table
    operator/(const canonical_table& f, const canonical_table& g) {
      return join<canonical_table>(f, g, std::minus<T>());
    }

    //! Multiplies a canonical_table factor by a constant.
    friend canonical_table
    operator*(const canonical_table& f, logarithmic<T> x) {
      return transform<canonical_table>(f, incremented_by<T>(x.lv));
    }

    //! Multiplies a canonical_table factor by a constant.
    friend canonical_table
    operator*(logarithmic<T> x, const canonical_table& f) {
      return transform<canonical_table>(f, incremented_by<T>(x.lv));
    }

    //! Divides a canonical_table factor by a constant.
    friend canonical_table
    operator/(const canonical_table& f, logarithmic<T> x) {
      return transform<canonical_table>(f, decremented_by<T>(x.lv));
    }

    //! Divides a constant by a canonical_table factor.
    friend canonical_table
    operator/(logarithmic<T> x, const canonical_table& f) {
      return transform<canonical_table>(f, subtracted_from<T>(x.lv));
    }

    //! Raises the canonical_table factor by an exponent.
    friend canonical_table
    pow(const canonical_table& f, T x) {
      return transform<canonical_table>(f, multiplied_by<T>(x));
    }

    //! Element-wise maximum of two factors.
    friend canonical_table
    max(const canonical_table& f, const canonical_table& g) {
      return transform<canonical_table>(f, g, libgm::maximum<T>());
    }

    //! Element-wise minimum of two factors.
    friend canonical_table
    min(const canonical_table& f, const canonical_table& g) {
      return transform<canonical_table>(f, g, libgm::minimum<T>());
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend canonical_table
    weighted_update(const canonical_table& f, const canonical_table& g, T a) {
      return transform<canonical_table>(f, g, weighted_plus<T>(1 - a, a));
    }

    //! Computes the marginal of the factor over a subset of variables.
    canonical_table marginal(const domain_type& retain) const {
      canonical_table result;
      marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    canonical_table maximum(const domain_type& retain) const {
      canonical_table result;
      maximum(retain, result);
      return result;
    }

    //! Computes the minimum for each assignment to the given variables.
    canonical_table minimum(const domain_type& retain) const {
      canonical_table result;
      minimum(retain, result);
      return result;
    }

    //! If this factor represents p(x, y), returns p(x | y).
    canonical_table conditional(const domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const domain_type& retain, canonical_table& result) const {
      T offset = maximum().lv;
      this->aggregate(retain, T(0), plus_exponent<T>(-offset), result);
      for (T& x : result.param_) { x = std::log(x) + offset; }
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const domain_type& retain, canonical_table& result) const {
      this->aggregate(retain, -inf<T>(), libgm::maximum<T>(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const domain_type& retain, canonical_table& result) const {
      this->aggregate(retain, +inf<T>(), libgm::minimum<T>(), result);
    }

    //! Returns the normalization constant of the factor.
    logarithmic<T> marginal() const {
      T offset = maximum().lv;
      T sum = this->param_.accumulate(T(0), plus_exponent<T>(-offset));
      return logarithmic<T>(std::log(sum) + offset, log_tag());
    }

    //! Returns the maximum value in the factor.
    logarithmic<T> maximum() const {
      T result = this->param_.accumulate(-inf<T>(), libgm::maximum<T>());
      return logarithmic<T>(result, log_tag());
    }

    //! Returns the minimum value in the factor.
    logarithmic<T> minimum() const {
      T result = this->param_.accumulate(+inf<T>(), libgm::minimum<T>());
      return logarithmic<T>(result, log_tag());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    logarithmic<T> maximum(assignment_type& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      a.insert_or_assign(arguments(), this->param_.index(it));
      return logarithmic<T>(*it, log_tag());
    }

    //! Computes the minimum value and stores the corresponding assignment.
    logarithmic<T> minimum(assignment_type& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      a.insert_or_assign(arguments(), this->param_.index(it));
      return logarithmic<T>(*it, log_tag());
    }

    //! Normalizes the factor in-place.
    canonical_table& normalize() {
      this->param_ -= marginal().lv;
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return std::isfinite(maximum().lv);
    }

    //! Restricts this factor to an assignment.
    canonical_table restrict(const assignment_type& a) const {
      canonical_table result;
      restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const assignment_type& a, canonical_table& result) const {
      table_factor<Arg, T>::restrict(a, result);
    }

    // Sampling
    //==========================================================================

    //! Returns the distribution with the parameters of this factor.
    multivariate_categorical_distribution<T> distribution() const {
      return { this->param_, log_tag() };
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    uint_vector sample(Generator& rng) const {
      return sample(rng, uint_vector());
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    uint_vector sample(Generator& rng, const uint_vector& tail) const {
      return this->param_.sample(exponent<T>(), rng, tail);
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
                                               entropy_log_op<T>(),
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
    friend T cross_entropy(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, entropy_log_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, kld_log_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, jsd_log_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }

    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const canonical_table& p, const canonical_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), libgm::maximum<T>());
    }

  }; // class canonical_table

  // Input / output
  //============================================================================

  /**
   * Prints a human-readable representatino of the table factor to the stream.
   * \relates canonical_table
   */
  template <typename Arg, typename T>
  std::ostream&
  operator<<(std::ostream& out, const canonical_table<Arg, T>& f) {
    out << "#CT(" << f.arguments() << ")" << std::endl;
    out << f.param();
    return out;
  }

} // namespace libgm

#endif
