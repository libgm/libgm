#ifndef LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP
#define LIBGM_FACTOR_LOGARITHMIC_MATRIX_HPP

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
#include <libgm/math/likelihood/LogarithmicMatrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicMatrix<T>::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  DenseMatrix<T> param_;

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const LogarithmicMatrix<T>& other) override {
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

  LogarithmicMatrix<T> multiply(Exp<T> x) const {
    return param_ + x.lv;
  }

  LogarithmicMatrix<T> divide(Exp<T> x) const {
    return param_ - x.lv;
  }

  LogarithmicMatrix<T> divide_inverse(Exp<T> x) const {
    return x.lv - param_;
  }

  LogarithmicMatrix<T> pow(T x) const {
    return param_ * x;
  }

  LogarithmicMatrix<T> add(const LogarithmicMatrix<T>& other) const {
    // log_plus_exp;
  }

  LogarithmicMatrix<T> multiply(const LogarithmicMatrix<T>& other) const {
    return param_ + other.impl().param_;
  }

  LogarithmicMatrix<T> divide(const LogarithmicMatrix<T>& other) const {
    return param_ - other.impl().param_;
  }

  LogarithmicMatrix<T> weighted_update(const LogarithmicMatrix<T>& other, T x) const {
    return param_ * (1 - x) + other.impl().param_ * x;
  }

  // Join operations
  //--------------------------------------------------------------------------
  LogarithmicMatrix<T> multiply_front(const LogarithmicVector<T>& other) const {
    return param_.colwise() + other.param();
  }

  LogarithmicMatrix<T> multiply_back(const LogarithmicVector<T>& other) const {
    return param_.rowwise() + other.param().transpose();
  }

  LogarithmicMatrix<T> divide_front(const LogarithmicVector<T>& other) const {
    return param_.colwise() - other.param();
  }

  LogarithmicMatrix<T> divide_back(const LogarithmicVector<T>& other) const {
    return param_.rowwise() - other.param().transpose();
  }

  // Mutations
  //--------------------------------------------------------------------------
  void multiply_in(Exp<T> x) {
    param_.array() += x.lv;
  }

  void divide_in(Exp<T> x) {
    param_.array() -= x.lv;
  }

  void multiply_in(const LogarithmicMatrix<T>& other) {
    param_ += other.param_;
  }

  void divide_in(const LogarithmicMatrix& other){
    param_ -= other.param_;
  }

  void normalize() {
    *this /= marginal();
  }

  void multiply_in_front(const LogarithmicVector<T>& other) {
    param_.colwise() *= other.param();
  }

  void multiply_in_back(const LogarithmicVector<T>& other) {
    param_.rowwise() *= other.param().transpose();
  }

  void divide_in_front(const LogarithmicVector<T>& other) {
    param_.colwise() /= other.param();
  }

  void divide_in_back(const LogarithmicVector<T>& other) {
    param_.rowwise() /= other.param().transpose();
  }

  // Conversions
  //--------------------------------------------------------------------------

  ProbabilityMatrix<T> probability() const {
    return exp(param_);
  }

  /**
    * Returns a logarithmic_table expression equivalent to this matrix.
    */
  LogarithmicTable<T> table() const {
    return table_from_matrix<log_tag>(derived()); // in table_function.hpp
  }

  // Aggregates
  //--------------------------------------------------------------------------

  LogarithmicVector<T> marginal_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().logSumExp();
  }

  LogarithmicVector<T> marginal_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().logSumExp();
  }

  LogarithmicVector<T> maximum_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().logSumExp();roderived().aggregate(member_maxCoeff(), retain);
  }

  LogarithmicVector<T> maximum_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().logSumExp();roderived().aggregate(member_maxCoeff(), retain);
  }

  LogarithmicVector<T> minimum_front(size_t n) const {
    assert(n == 1);
    return param_.rowwise().logSumExp();roderived().aggregate(member_minCoeff(), retain);
  }

  LogarithmicVector<T> minimum_back(size_t n) const {
    assert(n == 1);
    return param_.colwise().logSumExp();roderived().aggregate(member_minCoeff(), retain);
  }
  Exp<T> marginal() const {
    return Exp<T>(log_sum_exp...);
  }

  Exp<T> maximum() const {
    return Exp<T>(*std::max_element(begin(), end()));
  }

  Exp<T> minimum() const {
    return Exp<T>(*std::min_element(begin(), end()));
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
    return max().lv > -inf<T>();
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
  LogarithmicMatrix<T> conditional(size_t nhead = 1) const {
    assert(nhead == 1);
    DenseMatrix<T> result;
    result.array().rowise() -= member_logSumExp()(result.array().colwise());
    return std::move(restult);
  }

  LogarithmicVector<T> restrict_head(size_t n, const Assignment& a) const {
    assert(n == 1);
    return param_.row(a.get<size_t>(0));
  }

  LogarithmicVector<T> restrict_tail(size_t n, const Assignment& a) const {
    assert(n = 1);
    return param_.col(a.get<size_t>(0));
  }

  // Reshaping
  //--------------------------------------------------------------------------

  /**
   * Returns the expression representing the transpose of this expression.
   */
  LogarithmicMatrix<T> transpose() const {
    return param_.transpose();
  }

  // Sampling
  //--------------------------------------------------------------------------

  /**
    * Returns a categorical distribution represented by this expression.
    */
  BivariateCategoricalDistribution<T> distribution() const {
    return { param_, log_tag() };
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
    RealType p = std::uniform_real_distribution<RealType>()(rng);
    return derived().find_if(
      compose(partial_sum_greater_than<RealType>(p), exponent<RealType>())
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
    auto plus_entropy = compose_right(std::plus<RealType>(), EntropyLogOp<RealType>());
    return std::accumulate(param_.data(), param_.data() + param_.size(), T(0), plus_entropy);
  }

  T cross_entropy(const LogarithmicMatrix<T>& other) const {
    return transform_accumulate(EntropyLogOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T kl_divergence(const LogarithmicMatrix<T>& other) const {
    return transform_accumulate(KldLogOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T sum_diff(const LogarithmicMatrix<T>& other) const {
    return (param_ - other.param_).abs().sum();
  }

  T max_diff(const LogarithmicMatrix<T>& other) const {
    return (param_ - other.param_).abs().maxCoeff();
  }
}; // Impl

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols) {
  reset(rows, cols);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(std::pair<size_t, size_t> shape) {
  reset(shape.first, shape.second);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x) {
  reset(rows, cols);
  impl().param_.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(std::pair<size_t, size_t> shape, Exp<T> x) {
  reset(shape.first, shape.second);
  impl().param_.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(const DenseMatrix<T>& param) {
  impl_.reset(new Impl(param));
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(DenseMatrix<T>&& param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values) {
  assert(values.size() == rows * cols);
  reset(rows, cols);
  std::copy(values.begin(), values.end(), impl().param_.data());
}

template <typename T>
void LogarithmicMatrix<T>::reset(size_t rows, size_t cols) {
  if (impl_) {
    impl().data_.resize(rows, cols);
  } else {
    impl_.reset(new Impl(rows, cols));
  }
}

template <typename T>
size_t LogarithmicMatrix<T>::rows() const {
  return impl().param_.rows();
}

template <typename T>
size_t LogarithmicMatrix<T>::cols() const {
  return impl().param_.cols();
}

template <typename T>
size_t LogarithmicMatrix<T>::size() const {
  return impl().param_.size();
}

template <typename T>
T* LogarithmicMatrix<T>::begin() {
  return impl().param_.data();
}

template <typename T>
const T* LogarithmicMatrix<T>::begin() const {
  return impl().param_.data();
}

template <typename T>
T* LogarithmicMatrix<T>::end() {
  return begin() + size();
}

template <typename T>
const T* LogarithmicMatrix<T>::end() const {
  return begin() + size();
}

template <typename T>
DenseMatrix<T>& LogarithmicMatrix<T>::param() {
  return impl().param_;
}

template <typename T>
const DenseMatrix<T>& LogarithmicMatrix<T>::param() const {
  return impl().param_;
}

template <typename T>
T LogarithmicMatrix<T>::log(size_t row, size_t col) const {
  return impl().param_(row, col);
}

template <typename T>
T LogarithmicMatrix<T>::log(const Assignment& a) const {
  return impl().param_(a.get<size_t>(0), a.get<size_t>(1));
}

}; // class LogarithmicMatrix

} // namespace libgm

#endif
