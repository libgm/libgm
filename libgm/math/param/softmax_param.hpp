#ifndef LIBGM_SOFTMAX_PARAM_HPP
#define LIBGM_SOFTMAX_PARAM_HPP

#include <libgm/datastructure/hybrid_index.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/serialization/eigen.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace libgm {

  /**
   * A softmax function over one discrete variable y and a vector of
   * real-valued features x. This function is equal to a normalized
   * exponential, f(y=i, x) = exp(b_i + w_i^T x) / sum_j exp(b_j + w_j^T x).
   * Here, b is a bias vector and w is a weight matrix with rows w_i^T.
   * The parameter matrices are dense, but the function can be evaluated
   * on sparse feature vectors.
   *
   * This class models the OptimizationVector concept and can be directly
   * used in optimization classes.
   *
   * \tparam T a real type for representing each parameter.
   * \ingroup math_functions
   */
  template <typename T>
  class softmax_param {
  public:
    // Public types
    //======================================================================
    // OptimizationVector types
    typedef T value_type;

    // Underlying representation
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    // Constructors
    //======================================================================

    /**
     * Creates an empty softmax. This does not represent a valid function.
     */
    softmax_param() { }

    /**
     * Creates a softmax function with the given number of labels and
     * features. Allocates the parameters, but does not initialize them
     * to any specific value.
     */
    softmax_param(std::size_t labels, std::size_t features)
      : weight_(labels, features), bias_(labels) { }

    /**
     * Creates a softmax function with the given number of labels and
     * features, and initializes the parameters to the given value.
     */
    softmax_param(std::size_t labels, std::size_t features, T init)
      : weight_(labels, features), bias_(labels) {
      bias_.fill(init);
      weight_.fill(init);
    }

    /**
     * Creates a softmax function with the given parameters.
     */
    softmax_param(const mat_type& weight, const vec_type& bias)
      : weight_(weight), bias_(bias) {
      assert(weight_.rows() == bias_.rows());
    }

    /**
     * Creates a softmax function with the given parameters.
     */
    softmax_param(mat_type&& weight, vec_type&& bias) {
      weight_.swap(weight);
      bias_.swap(bias);
      assert(weight_.rows() == bias_.rows());
    }

    //! Copy constructor.
    softmax_param(const softmax_param& other) = default;

    //! Move constructor.
    softmax_param(softmax_param&& other) {
      swap(*this, other);
    }

    //! Assignment operator.
    softmax_param& operator=(const softmax_param& other) {
      if (this != &other) {
        weight_ = other.weight_;
        bias_ = other.bias_;
      }
      return *this;
    }

    //! Move assignment operator.
    softmax_param& operator=(softmax_param&& other) {
      swap(*this, other);
      return *this;
    }

    //! Swaps the content of two softmax functions.
    friend void swap(softmax_param& f, softmax_param& g) {
      f.weight_.swap(g.weight_);
      f.bias_.swap(g.bias_);
    }

    //! Serializes the parameters to an archive.
    void save(oarchive& ar) const {
      ar << weight_ << bias_;
    }

    //! Deserializes the parameters from an archive.
    void load(iarchive& ar) {
      ar >> weight_ >> bias_;
      assert(weight_.rows() == bias_.rows());
    }

    /**
     * Resets the function to the given number of labels and features.
     * May invalidate the parameters.
     */
    void resize(std::size_t labels, std::size_t features) {
      weight_.resize(labels, features);
      bias_.resize(labels);
    }

    /**
     * Sets the function to the given number of labels and features,
     * filling the contents with 0.
     */
    void zero(std::size_t labels, std::size_t features) {
      weight_.setZero(labels, features);
      bias_.setZero(labels);
    }

    /**
     * Fills the parameters with the given constant.
     */
    void fill(T value) {
      weight_.fill(value);
      bias_.fill(value);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns true if the softmax function is empty.
    bool empty() const {
      return !weight_.data();
    }

    //! Returns the number of labels.
    std::size_t labels() const {
      return weight_.rows();
    }

    //! Returns the number of features.
    std::size_t features() const {
      return weight_.cols();
    }

    //! Returns the weight matrix.
    mat_type& weight() {
      return weight_;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      return weight_;
    }

    //! Returns the bias vector.
    vec_type& bias() {
      return bias_;
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      return bias_;
    }

    //! Returns the weight with the given indices.
    T& weight(std::size_t i, std::size_t j) {
      return weight_(i, j);
    }

    //! Returns the weight with the given indices.
    const T& weight(std::size_t i, std::size_t j) const {
      return weight_(i, j);
    }

    //! Returns the bias with the given index.
    T& bias(std::size_t i) {
      return bias_[i];
    }

    //! Returns the bias with the given index.
    const T& bias(std::size_t i) const {
      return bias_[i];
    }

    //! Evaluates the function for an Eigen dense or sparse vector.
    template <typename Derived>
    vec_type operator()(const Eigen::EigenBase<Derived>& x) const {
      assert(x.rows() == weight_.cols());
      assert(x.cols() == 1);
      vec_type y(weight_ * x.derived() + bias_);
      y = y.array().exp();
      y /= y.sum();
      return y;
    }

    //! Evaluates the function for a sparse feature vector with unit values.
    vec_type operator()(const std::vector<std::size_t>& x) const {
      vec_type y = bias_;
      for (std::size_t i : x) {
        assert(i < weight_.cols());
        y += weight_.col(i);
      }
      y = y.array().exp();
      y /= y.sum();
      return y;
    }

    //! Returns the log-value for a dense feature vector.
    template <typename Derived>
    vec_type log(const Eigen::EigenBase<Derived>& x) const {
      vec_type y = operator()(x);
      y = y.array().log();
      return y;
    }

    //! Returns the log-value for a sparse feature vector.
    vec_type log(const std::vector<std::size_t>& x) const {
      vec_type y = operator()(x);
      y = y.array().log();
      return y;
    }

    //! Returns true if all the parameters are finite and not NaN.
    bool is_finite() const {
      return weight_.allFinite() && bias_.allFinite();
    }

    //! Returns true if two softmax parameter vectors are equal.
    friend bool operator==(const softmax_param& f, const softmax_param& g) {
      return f.weight_ == g.weight_ && f.bias_ == g.bias_;
    }

    //! Returns true if two softmax parameter vecors are not equal.
    friend bool operator!=(const softmax_param& f, const softmax_param& g) {
      return !(f == g);
    }

    // Sampling
    //==========================================================================
    template <typename Generator, typename Derived>
    std::size_t sample(Generator& rng,
                       const Eigen::EigenBase<Derived>& x) const {
      vec_type p = operator()(x);
      T val = std::uniform_real_distribution<T>()(rng);
      for (std::size_t i = 0; i < p.size(); ++i) {
        if (val <= p[i]) {
          return i;
        } else {
          val -= p[i];
        }
      }
      throw std::logic_error("The probabilities do not sum to 1");
    }

    // OptimizationVector functions
    //=========================================================================
    softmax_param operator-() const {
      return softmax_param(-weight_, -bias_);
    }

    softmax_param& operator+=(const softmax_param& f) {
      weight_ += f.weight_;
      bias_ += f.bias_;
      return *this;
    }

    softmax_param& operator-=(const softmax_param& f) {
      weight_ -= f.weight_;
      bias_ -= f.bias_;
      return *this;
    }

    softmax_param& operator/=(const softmax_param& f) {
      weight_.array() /= f.weight_.array();
      bias_.array() /= f.bias_.array();
      return *this;
    }

    softmax_param& operator+=(T a) {
      weight_.array() += a;
      bias_.array() += a;
      return *this;
    }

    softmax_param& operator-=(T a) {
      weight_.array() -= a;
      bias_.array() -= a;
      return *this;
    }

    softmax_param& operator*=(T a) {
      weight_ *= a;
      bias_ *= a;
      return *this;
    }

    softmax_param& operator/=(T a) {
      weight_ /= a;
      bias_ /= a;
      return *this;
    }

    friend void copy_shape(const softmax_param& src, softmax_param& dst) {
      dst.resize(src.labels(), src.features());
    }

    friend void update(softmax_param& f, const softmax_param& g, T a) {
      f.weight_ += a * g.weight_;
      f.bias_ += a * g.bias_;
    }

    friend T dot(const softmax_param& f, const softmax_param& g) {
      return f.weight_.cwiseProduct(g.weight_).sum() + f.bias_.dot(g.bias_);
    }

  private:
    // Private members
    //=========================================================================

    //! The weight matrix.
    mat_type weight_;

    //! The bias vector.
    vec_type bias_;

  }; // class softmax_param

  /**
   * Prints the softmax function parameters to a stream.
   * \relates softmax_param
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const softmax_param<T>& f) {
    typename softmax_param<T>::mat_type a(f.labels(), f.features() + 1);
    a << f.weight(), f.bias();
    out << a << std::endl;
    return out;
  }

} // namespace libgm

#endif
