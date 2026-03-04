#pragma once

#include "../probability_matrix.hpp"

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
// #include <libgm/math/likelihood/ProbabilityMatrix_ll.hpp>
// #include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct ProbabilityMatrix<T>::Impl : Object::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(param);
  }

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  explicit Impl(size_t rows, size_t cols)
    : param(rows, cols) {}

  explicit Impl(const Shape& shape) {
    assert(shape.size() == 2);
    param.resize(shape[0], shape[1]);
  }

  explicit Impl(Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param)
    : param(std::move(param)) {}

  // Utility functions
  //--------------------------------------------------------------------------

  const T* begin() const {
    return param.data();
  }

  const T* end() const {
    return param.data() + param.size();
  }

  // Object functions
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    out << param;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply(const T& x, ProbabilityMatrix& result) const {
    result.param() = param * x;
  }

  void divide(const T& x, ProbabilityMatrix& result) const {
    result.param() = param / x;
  }

  void divide_inverse(const T& x, ProbabilityMatrix& result) const {
    result.param() = x * param;
  }

  void multiply(const ProbabilityMatrix& other, ProbabilityMatrix& result) const {
    result.param() = param * other.param();
  }

  void divide(const ProbabilityMatrix& other, ProbabilityMatrix& result) const {
    result.param() = param / other.param();
  }

  void multiply_in(const T& x) {
    param *= x;
  }

  void divide_in(const T& x) {
    param /= x;
  }

  void multiply_in(const ProbabilityMatrix& other) {
    param *= other.param();
  }

  void divide_in(const ProbabilityMatrix& other){
    param /= other.param();
  }

  // Join operations
  //--------------------------------------------------------------------------

  void multiply_front(const ProbabilityVector<T>& other, ProbabilityMatrix& result) const {
    result.param() = param.colwise() * other.param();
  }

  void multiply_back(const ProbabilityVector<T>& other, ProbabilityMatrix& result) const {
    result.param() = param.rowwise() * other.param().transpose();
  }

  void divide_front(const ProbabilityVector<T>& other, ProbabilityMatrix& result) const {
    result.param() = param.colwise() / other.param();
  }

  void divide_back(const ProbabilityVector<T>& other, ProbabilityMatrix& result) const {
    result.param() = param.rowwise() / other.param().transpose();
  }

  void multiply_in_front(const ProbabilityVector<T>& other) {
    param.colwise() *= other.param();
  }

  void multiply_in_back(const ProbabilityVector<T>& other) {
    param.rowwise() *= other.param().transpose();
  }

  void divide_in_front(const ProbabilityVector<T>& other) {
    param.colwise() /= other.param();
  }

  void divide_in_back(const ProbabilityVector<T>& other) {
    param.rowwise() /= other.param().transpose();
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  void power(T x, ProbabilityMatrix& result) const {
    result.param() = param.pow(x);
  }

  void weighted_update(const ProbabilityMatrix& other, T x, ProbabilityMatrix& result) const {
    result.param() = param * (1 - x) + other.param() * x;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param.sum();
  }

  T maximum(std::vector<size_t>* values) const {
    if (values) {
      values->resize(2);
      size_t* data = values->data();
      return param.maxCoeff(data, data + 1);
    } else {
      return param.maxCoeff();
    }
  }

  T minimum(std::vector<size_t>* values) const {
    if (values) {
      values->resize(2);
      size_t* data = values->data();
      return param.minCoeff(data, data + 1);
    } else {
      return param.minCoeff();
    }
  }

  void marginal_front(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.rowwise().sum();
  }

  void marginal_back(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.colwise().sum().transpose();
  }

  void maximum_front(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.rowwise().maxCoeff();
  }

  void maximum_back(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.colwise().maxCoeff().transpose();
  }

  void minimum_front(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.rowwise().minCoeff();
  }

  void minimum_back(unsigned n, ProbabilityVector<T>& result) const {
    assert(n == 1);
    result.param() = param.colwise().minCoeff().transpose();
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    divide_in(marginal());
  }

  void normalize(unsigned nhead) {
    assert(nhead == 1);
    param.rowwise() /= param.colwise().sum();
  }

  // Restrictions
  //--------------------------------------------------------------------------

  void restrict_front(const std::vector<size_t>& values, ProbabilityVector<T>& result) const {
    result.param() = param.row(values[0]).transpose();
  }

  void restrict_back(const std::vector<size_t>& values, ProbabilityVector<T>& result) const {
    result.param() = param.col(values[0]);
  }

  // Reshaping
  //--------------------------------------------------------------------------

  /**
   * Returns the expression representing the transpose of this expression.
   */
  void transpose(ProbabilityMatrix& result) const {
    result.param() = param.transpose();
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T acc, T val) {
      return acc + EntropyOp<T>()(val);
    });
  }

  template <typename Op>
  T transform_sum(const ProbabilityMatrix& other, Op op) const {
    assert(param.rows() == other.rows() && param.cols() == other.cols());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), std::plus<T>(), op);
  }

  T cross_entropy(const ProbabilityMatrix& other) const {
    return transform_sum(other, EntropyOp<T>());
  }

  T kl_divergence(const ProbabilityMatrix& other) const {
    return transform_sum(other, KldOp<T>());
  }

  T sum_difference(const ProbabilityMatrix& other) const {
    return (param - other.param()).abs().sum();
  }

  T max_difference  (const ProbabilityMatrix& other) const {
    return (param - other.param()).abs().maxCoeff();
  }

#if 0
  // Sampling
  //--------------------------------------------------------------------------

  /**
    * Returns a categorical distribution represented by this expression.
    */
  BivariateCategoricalDistribution<T> distribution() const {
    return param;
  }

  /**
   * Draws a random sample from a marginal distribution represented by this
   * expression.
   *
   * \throw std::out_of_range
   *        may be thrown if the distribution is not normalized
   */
  template <typename Generator>
  std::pair<size_t, size_t> sample(Generator& rng) const {
    T p = std::uniform_real_distribution<RealType>()(rng);
    return derived().find_if(partial_sum_greater_than<T>(p));
    );
  }

  /**
    * Draws a random sample from a marginal distribution represented by this
    * expression, storing the result in an output vector.
    *
    * \throw std::out_of_range
    *        may be thrown if the distribution is not normalized
    */
  template <typename Generator>
  void sample(Generator& rng, uint_vector& result) const {
    result.resize(2);
    std::tie(result.front(), result.back()) = sample(rng);
  }
#endif

}; // class Impl

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(size_t rows, size_t cols, T x)
  : Object(std::make_unique<Impl>(rows, cols)) {
  impl().param.fill(x);
}

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(const Shape& shape, T x)
  : Object(std::make_unique<Impl>(shape)) {
  impl().param.fill(x);
}

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(size_t rows, size_t cols, std::initializer_list<T> values)
  : Object(std::make_unique<Impl>(rows, cols)) {
  assert(values.size() == rows * cols);
  std::copy(values.begin(), values.end(), impl().param.data());
}

template <typename T>
size_t ProbabilityMatrix<T>::rows() const {
  return impl().param.rows();
}

template <typename T>
size_t ProbabilityMatrix<T>::cols() const {
  return impl().param.cols();
}

template <typename T>
size_t ProbabilityMatrix<T>::size() const {
  return impl().param.size();
}

template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& ProbabilityMatrix<T>::param() {
  return impl().param;
}

template <typename T>
const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& ProbabilityMatrix<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbabilityMatrix<T>::operator()(size_t row, size_t col) const {
  return impl().param(row, col);
}

template <typename T>
T ProbabilityMatrix<T>::operator()(const std::vector<size_t>& values) const {
  assert(values.size() == 2);
  return impl().param(values[0], values[1]);
}

template <typename T>
T ProbabilityMatrix<T>::log(size_t row, size_t col) const {
  return std::log(impl().param(row, col));
}

template <typename T>
T ProbabilityMatrix<T>::log(const std::vector<size_t>& values) const {
  return std::log(impl().param(values[0], values[1]));
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator*(T x) const {
  ProbabilityMatrix result;
  impl().multiply(x, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator*(const ProbabilityMatrix& other) const {
  ProbabilityMatrix result;
  impl().multiply(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator*=(T x) {
  impl().multiply_in(x);
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator*=(const ProbabilityMatrix& other) {
  impl().multiply_in(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator/(T x) const {
  ProbabilityMatrix result;
  impl().divide(x, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_inverse(T x) const {
  ProbabilityMatrix result;
  impl().divide_inverse(x, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator/(const ProbabilityMatrix& other) const {
  ProbabilityMatrix result;
  impl().divide(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator/=(T x) {
  impl().divide_in(x);
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator/=(const ProbabilityMatrix& other) {
  impl().divide_in(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::multiply_front(const ProbabilityVector<T>& other) const {
  ProbabilityMatrix result;
  impl().multiply_front(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::multiply_back(const ProbabilityVector<T>& other) const {
  ProbabilityMatrix result;
  impl().multiply_back(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::multiply_in_front(const ProbabilityVector<T>& other) {
  impl().multiply_in_front(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::multiply_in_back(const ProbabilityVector<T>& other) {
  impl().multiply_in_back(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_front(const ProbabilityVector<T>& other) const {
  ProbabilityMatrix result;
  impl().divide_front(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_back(const ProbabilityVector<T>& other) const {
  ProbabilityMatrix result;
  impl().divide_back(other, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::divide_in_front(const ProbabilityVector<T>& other) {
  impl().divide_in_front(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::divide_in_back(const ProbabilityVector<T>& other) {
  impl().divide_in_back(other);
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::pow(T x) const {
  ProbabilityMatrix result;
  impl().power(x, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::weighted_update(const ProbabilityMatrix& other, T x) const {
  ProbabilityMatrix result;
  impl().weighted_update(other, x, result);
  return result;
}

template <typename T>
T ProbabilityMatrix<T>::marginal() const {
  return impl().marginal();
}

template <typename T>
T ProbabilityMatrix<T>::maximum(std::vector<size_t>* values) const {
  return impl().maximum(values);
}

template <typename T>
T ProbabilityMatrix<T>::minimum(std::vector<size_t>* values) const {
  return impl().minimum(values);
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::marginal_front(unsigned n) const {
  ProbabilityVector<T> result;
  impl().marginal_front(n, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::marginal_back(unsigned n) const {
  ProbabilityVector<T> result;
  impl().marginal_back(n, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::maximum_front(unsigned n) const {
  ProbabilityVector<T> result;
  impl().maximum_front(n, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::maximum_back(unsigned n) const {
  ProbabilityVector<T> result;
  impl().maximum_back(n, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::minimum_front(unsigned n) const {
  ProbabilityVector<T> result;
  impl().minimum_front(n, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::minimum_back(unsigned n) const {
  ProbabilityVector<T> result;
  impl().minimum_back(n, result);
  return result;
}

template <typename T>
void ProbabilityMatrix<T>::normalize() {
  impl().normalize();
}

template <typename T>
void ProbabilityMatrix<T>::normalize_head(unsigned nhead) {
  impl().normalize(nhead);
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::restrict_front(const std::vector<size_t>& values) const {
  ProbabilityVector<T> result;
  impl().restrict_front(values, result);
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::restrict_back(const std::vector<size_t>& values) const {
  ProbabilityVector<T> result;
  impl().restrict_back(values, result);
  return result;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::transpose() const {
  ProbabilityMatrix result;
  impl().transpose(result);
  return result;
}

template <typename T>
T ProbabilityMatrix<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T ProbabilityMatrix<T>::cross_entropy(const ProbabilityMatrix& other) const {
  return impl().cross_entropy(other);
}

template <typename T>
T ProbabilityMatrix<T>::kl_divergence(const ProbabilityMatrix& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
T ProbabilityMatrix<T>::sum_diff(const ProbabilityMatrix& other) const {
  return impl().sum_difference(other);
}

template <typename T>
T ProbabilityMatrix<T>::max_diff(const ProbabilityMatrix& other) const {
  return impl().max_difference(other);
}

template <typename T>
LogarithmicMatrix<T> ProbabilityMatrix<T>::logarithmic() const {
  return param().log();
}

template <typename T>
ProbabilityTable<T> ProbabilityMatrix<T>::table() const {
  return {{rows(), cols()}, param().data()};
}

template <typename T>
typename ProbabilityMatrix<T>::Impl& ProbabilityMatrix<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *static_cast<Impl*>(impl_.get());
}

template <typename T>
const typename ProbabilityMatrix<T>::Impl& ProbabilityMatrix<T>::impl() const {
  assert(impl_);
  return *static_cast<const Impl*>(impl_.get());
}

} // namespace libgm
