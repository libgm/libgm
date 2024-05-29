#pragma once

#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/ProbablityMatrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

/**
 * The base class for ProbablityMatrix factors and expressions.
 *
 * \tparam RealType
 *         The real type representing the parameters.
 * \tparam Derived
 *         The expression type that derives from this base class.
 *         The type must implement the following functions:
 *         alias(), eval_to().
 */
template <typename T>
struct ProbablityMatrix<T>::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  DenseMatrix<T> param_;

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const ProbablityMatrix<T>& other) override {
    return param_ == other.impl().param_;
  }

  void print(std::ostream& out) override {
    out << param_;
  }

  void save(oarchive& ar) const override {
    ar << param_;
  }

  void load(iarchive& ar) override {
    ar >> param_;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ProbablityMatrix<T> multiply(T x) const {
    return param_ * x;
  }

  ProbablityMatrix<T> divide(T x) const {
    return param_ / x;
  }

  ProbablityMatrix<T> divide_inverse(T x) const {
    return x * param_;
  }

  ProbablityMatrix<T> pow(T x) const {
    return param_.pow(x);
  }

  ProbablityMatrix<T> add(const ProbablityMatrix<T>& other) const {
    return param_ + other.param();
  }

  ProbablityMatrix<T> multiply(const ProbablityMatrix<T>& other) const {
    return param_ * other.impl().param_;
  }

  ProbablityMatrix<T> divide(const ProbablityMatrix<T>& other) const {
    return param_ / other.impl().param_;
  }

  ProbablityMatrix<T> weighted_update(const ProbablityMatrix<T>& other, T x) const {
    return param_ * (1 - x) + other.impl().param_ * x;
  }

  // Join operations
  //--------------------------------------------------------------------------
  ProbablityMatrix<T> multiply_front(const ProbablityVector<T>& other) const {
    return param_.colwise() * other.param();
  }

  ProbablityMatrix<T> multiply_back(const ProbablityVector<T>& other) const {
    return param_.rowwise() * other.param().transpose();
  }

  ProbablityMatrix<T> divide_front(const ProbablityVector<T>& other) const {
    return param_.colwise() / other.param();
  }

  ProbablityMatrix<T> divide_back(const ProbablityVector<T>& other) const {
    return param_.rowwise() / other.param().transpose();
  }

  // Mutations
  //--------------------------------------------------------------------------
  void multiply_in(T x) {
    param_.array() *= x;
  }

  void divide_in(T x) {
    param_.array() /= x;
  }

  void multiply_in(const ProbablityMatrix<T>& other) {
    param_ *= other.param_;
  }

  void divide_in(const ProbablityMatrix& other){
    param_ /= other.param_;
  }

  void normalize() {
    *this /= marginal();
  }

  void multiply_in_front(const ProbablityVector<T>& other) {
    param_.colwise() *= other.param();
  }

  void multiply_in_back(const ProbablityVector<T>& other) {
    param_.rowwise() *= other.param().transpose();
  }

  void divide_in_front(const ProbablityVector<T>& other) {
    param_.colwise() /= other.param();
  }

  void divide_in_back(const ProbablityVector<T>& other) {
    param_.rowwise() /= other.param().transpose();
  }

  // Conversions
  //--------------------------------------------------------------------------

  LogarithmicMatrix<T> logarithmic() const {
    return log(param_);
  }

  /**
    * Returns a logarithmic_table expression equivalent to this matrix.
    */
  ProbablityTable<T> table() const {
    return table_from_matrix(derived()); // in table_function.hpp
  }

  // Aggregates
  //--------------------------------------------------------------------------

  ProbablityVector<T> marginal_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().sum();
  }

  ProbablityVector<T> marginal_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().sum();
  }

  ProbablityVector<T> maximum_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().maxCoeff();
  }

  ProbablityVector<T> maximum_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().maxCoeff();
  }

  ProbablityVector<T> minimum_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().minCoeff();
  }

  ProbablityVector<T> minimum_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().minCoeff();
  }

  T marginal() const {
    return param_.sum();
  }

  T maximum() const {
    return *std::max_element(begin(), end());
  }

  T minimum() const {
    return *std::min_element(begin(), end());
  }

  Exp<T> maximum(size_t& row, size_t& col) const {
    auto it = std::max_element(begin(), end());
    row = ...;
    col = ...;
    return *it;
  }

  Exp<T> minimum(size_t& row, size_t& col) const {
    auto it = std::min_element(begin(), end());
    row = ...;
    col = ...;
    return *it;
  }

  bool normalizable() const {
    return max() > T(0);
  }

  // Conditioning
  //--------------------------------------------------------------------------

  /**
    * If this expression represents a marginal distribution p(x, y), this
    * function returns a probability_matrix expression representing the
    * conditional p(x | y) with 1 tail (front) dimension.
    *
    * The optional argument must be always 1.
    */
  ProbablityMatrix<T> conditional(size_t nhead = 1) const {
    assert(nhead == 1);
    DenseMatrix<T> result;
    result.array().rowise() /= result.array().colwise().sum();
    return std::move(restult);
  }

  ProbablityVector<T> restrict_head(size_t n, const Assignment& a) const {
    assert(n == 1);
    return param_.row(a.get<size_t>(0));
  }

  ProbablityVector<T> restrict_tail(size_t n, const Assignment& a) const {
    assert(n = 1);
    return param_.col(a.get<size_t>(0));
  }

  // Reshaping
  //--------------------------------------------------------------------------

  /**
   * Returns the expression representing the transpose of this expression.
   */
  ProbablityMatrix<T> transpose() const {
    return param_.transpose();
  }

  // Sampling
  //--------------------------------------------------------------------------

  /**
    * Returns a categorical distribution represented by this expression.
    */
  BivariateCategoricalDistribution<T> distribution() const {
    return param_;
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

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    auto plus_entropy = compose_right(std::plus<RealType>(), EntropyOp<RealType>());
    return std::accumulate(param_.data(), param_.data() + param_.size(), T(0), plus_entropy);
  }

  T cross_entropy(const ProbablityMatrix<T>& other) const {
    return transform_accumulate(EntropyOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T kl_divergence(const ProbablityMatrix<T>& other) const {
    return transform_accumulate(KldOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T sum_diff(const ProbablityMatrix<T>& other) const {
    return (param_ - other.param_).abs().sum();
  }

  T max_diff(const ProbablityMatrix<T>& other) const {
    return (param_ - other.param_).abs().maxCoeff();
  }
}; // Impl

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols) {
  reset(rows, cols);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(std::pair<size_t, size_t> shape) {
  reset(shape.first, shape.second);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols, T x) {
  reset(rows, cols);
  impl().param_.fill(x);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(std::pair<size_t, size_t> shape, T x) {
  reset(shape.first, shape.second);
  impl().param_.fill(x);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(const DenseMatrix<T>& param) {
  impl_.reset(new Impl(param));
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(DenseMatrix<T>&& param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols, std::initializer_list<T> values) {
  assert(values.size() == rows * cols);
  reset(rows, cols);
  std::copy(values.begin(), values.end(), impl().param_.data());
}

template <typename T>
void ProbablityMatrix<T>::reset(size_t rows, size_t cols) {
  if (impl_) {
    impl().data_.resize(rows, cols);
  } else {
    impl_.reset(new Impl(rows, cols));
  }
}

template <typename T>
size_t ProbablityMatrix<T>::rows() const {
  return impl().param_.rows();
}

template <typename T>
size_t ProbablityMatrix<T>::cols() const {
  return impl().param_.cols();
}

template <typename T>
size_t ProbablityMatrix<T>::size() const {
  return impl().param_.size();
}

template <typename T>
T* ProbablityMatrix<T>::begin() {
  return impl().param_.data();
}

template <typename T>
const T* ProbablityMatrix<T>::begin() const {
  return impl().param_.data();
}

template <typename T>
T* ProbablityMatrix<T>::end() {
  return begin() + size();
}

template <typename T>
const T* ProbablityMatrix<T>::end() const {
  return begin() + size();
}

template <typename T>
DenseMatrix<T>& ProbablityMatrix<T>::param() {
  return impl().param_;
}

template <typename T>
const DenseMatrix<T>& ProbablityMatrix<T>::param() const {
  return impl().param_;
}

template <typename T>
T ProbablityMatrix<T>::operator()(size_t row, size_t col) const {
  return impl().param_(row, col);
}

template <typename T>
T ProbablityMatrix<T>::operator()(const Assignment& a) const {
  return impl().param_(a.get<size_t>(0), a.get<size_t>(1));
}

template <typename T>
T ProbablityMatrix<T>::log(size_t row, size_t col) const {
  return std::log(impl().param_(row, col));
}

template <typename T>
T ProbablityMatrix<T>::log(const Assignment& a) const {
  return std::log(impl().param_(a.get<size_t>(0), a.get<size_t>(1)));
}

}; // class ProbablityMatrix

} // namespace libgm

#endif
