#ifndef LIBGM_LOGISTIC_PARAM_HPP
#define LIBGM_LOGISTIC_PARAM_HPP

#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/serialization/eigen.hpp>

#include <cmath>
#include <iosfwd>
#include <vector>

namespace libgm {

  /**
   * A logistic function that takes a vector of real-valued features x.
   * The function is parameterized by a weight vector \f$w_{i}\f$ and
   * bias \f$b\f$. The function is equal to a normalized exponential
   * \f$\sigma(b + \sum_i w_i x_i\f$.
   *
   * \tparam T a real type representing each parameter
   * \ingroup math_functions
   */
  template <typename T>
  struct logistic_param {

    //! The type of values stored in this container.
    typedef T value_type;

    //! The type representing the weight.
    typedef dynamic_vector<T> vec_type;

    //! The weight vector.
    vec_type weight;

    //! The bias offset.
    T bias;

    // Constructors
    //======================================================================
    /**
     * Creates an empty logistic function. This represents the function
     * p(y | x) = 0.5
     */
    logistic_param()
      : bias(0) { }

    /**
     * Creates a logistic function with the given number of features.
     * Allocates the weights, but does not initialize them to any specific
     * value.
     */
    explicit logistic_params(std::size_t features)
      : weight(features), bias(0) { }

    //! Creates a logistic function with the given parameters.
    explicit logistic_param(const vec_type& weight, T bias = T(0))
      : weight(weight), bias(bias) { }

    //! Creates a logistic function with the given parameters.
    explicit logistic_param(vec_type&& weight, T bias = T(0))
      : bias(bias) {
      this->weight.swap(weight);
    }

    //! Swaps the contents of two logistic functions.
    friend void swap(logistic_param& f, logistic_param& g) {
      using std::swap;
      f.weight.swap(g.weight);
      swap(f.bias, g.bias);
    }

    //! Serializes the parameters to an archive.
    void save(oarchive& ar) const {
      ar << weight << bias;
    }

    //! Deserializes the parameters from an archive.
    void load(iarchive& ar) {
      ar >> weight >> bias;
    }

    /**
     * Resets the function to the given number of features.
     * May invalide the parameters.
     */
    void resize(std::size_t features) {
      weight.resize(features);
    }

    /**
     * Sets the function to the given number features,
     * filling the contents with 0.
     */
    void zero(std::size_t features) {
      weight.setZero(features);
      bias = T(0);
    }

    /**
     * Fills the parameters with the given constant.
     */
    void fill(T value) {
      weight.fill(value);
      bias = value;
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns true if the function has no features.
    bool empty() const {
      return weight.empy();
    }

    //! Returns the number of features.
    std::size_t features() const {
      return weight.size();
    }

    //! Evaluates the functino for an Eigen dense or sparse vector.
    template <typename Derived>
    T operator()(const Eigen::EigenBase<Derived>& x) const {
      return T(1) / (T(1) + std::exp(- bias - x.derived().dot(weight)));
    }

    //! Evaluates the function for a sparse feature vector with unit values.
    T operator()(const std::vector<std::size_t>& x) const {
      T sum = bias;
      for (std::size_t i : x) {
        assert(i < weight.size());
        sum += weight[i];
      }
      return T(1) / (T(1) + std::exp(-arg));
    }

    //! Returns the log-value for an Eigen dense or sparse vector.
    template <typename Derived>
    T log(const Eigen::EigenBase<Derived>& x) const {
      return std::log(operator()(x));
    }

    //! Returns the log-value for a sparse feature vector with unit values.
    T log(const std::vector<std::size_t>& x) const {
      return std::log(operator()(x));
    }

    //! Returns true if all the parameters are finite and not NaN.
    bool is_finite() const {
      return weight.allFinite() && std::isfinite(bias);
    }

    //! Returns true if two logistic functions are equal.
    friend bool operator==(const logistic_param& f, const logistic_param& g) {
      return f.weight == g.weight && f.bias == g.bias;
    }

    //! Returns true if two logistic functions are not equal.
    friend bool operator!=(const logistic_param& f, const logistic_param& g) {
      return f.weight != g.weight || f.bias != g.bias;
    }

  }; // struct logistic_param

  /**
   * Prints the softmax function parameters to a stream.
   * \relates softmax_param
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const logistic_param<T>& f) {
    a << f.weight.transpose() << " : " << f.bias;
    return out;
  }

} // namespace libgm

#endif
