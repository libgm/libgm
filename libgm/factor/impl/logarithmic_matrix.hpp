#pragma once

#include "../logarithmic_matrix.hpp"

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicMatrix<T>::Impl : Object::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(param);
  }

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  Impl(size_t rows, size_t cols)
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

  void multiply(const Exp<T>& x, LogarithmicMatrix& result) const {
    result.param() = param + x.lv;
  }

  void divide(const Exp<T>& x, LogarithmicMatrix& result) const {
    result.param() = param - x.lv;
  }

  void divide_inverse(const Exp<T>& x, LogarithmicMatrix& result) const {
    result.param() = x.lv - param;
  }

  void multiply(const LogarithmicMatrix& other, LogarithmicMatrix& result) const {
    result.param() = param + other.param();
  }

  void divide(const LogarithmicMatrix& other, LogarithmicMatrix& result) const {
    result.param() = param - other.param();
  }

  void multiply_in(const Exp<T>& x) {
    param.array() += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    param.array() -= x.lv;
  }

  void multiply_in(const LogarithmicMatrix& other) {
    param += other.param();
  }

  void divide_in(const LogarithmicMatrix& other){
    param -= other.param();
  }

  // Join operations
  //--------------------------------------------------------------------------

  void multiply_front(const LogarithmicVector<T>& other, LogarithmicMatrix& result) const {
    result.param() = param.colwise() + other.param();
  }

  void multiply_back(const LogarithmicVector<T>& other, LogarithmicMatrix& result) const {
    result.param() = param.rowwise() + other.param().transpose();
  }

  void divide_front(const LogarithmicVector<T>& other, LogarithmicMatrix& result) const {
    result.param() = param.colwise() - other.param();
  }

  void divide_back(const LogarithmicVector<T>& other, LogarithmicMatrix& result) const {
    result.param() = param.rowwise() - other.param().transpose();
  }

  void multiply_in_front(const LogarithmicVector<T>& other) {
    param.colwise() *= other.param();
  }

  void multiply_in_back(const LogarithmicVector<T>& other) {
    param.rowwise() *= other.param().transpose();
  }

  void divide_in_front(const LogarithmicVector<T>& other) {
    param.colwise() /= other.param();
  }

  void divide_in_back(const LogarithmicVector<T>& other) {
    param.rowwise() /= other.param().transpose();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  void power(T x, LogarithmicMatrix& result) const {
    result.param() = param * x;
  }

  void weighted_update(const LogarithmicMatrix& other, T x, LogarithmicMatrix& result) const {
    result.param() = param * (1 - x) + other.param() * x;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(DiscreteValues* values) const {
    if (values) {
      size_t* data = values->resize(2);
      return Exp<T>(param.maxCoeff(data, data + 1));
    } else {
      return Exp<T>(param.maxCoeff());
    }
  }

  Exp<T> minimum(DiscreteValues* values) const {
    if (values) {
      size_t* data = values->resize(2);
      return Exp<T>(param.minCoeff(data, data + 1));
    } else {
      return Exp<T>(param.minCoeff());
    }
  }

  void maximum_front(unsigned n, LogarithmicVector<T>& result) const {
    assert(n == 1);
    result.param() = param.rowwise().maxCoeff();
  }

  void maximum_back(unsigned n, LogarithmicVector<T>& result) const {
    assert(n == 1);
    result.param() = param.colwise().maxCoeff().transpose();
  }

  void minimum_front(unsigned n, LogarithmicVector<T>& result) const {
    assert(n == 1);
    result.param() = param.rowwise().minCoeff();
  }

  void minimum_back(unsigned n, LogarithmicVector<T>& result) const {
    assert(n == 1);
    result.param() = param.colwise().minCoeff().transpose();
  }

  // Restrictions
  //--------------------------------------------------------------------------

  void restrict_front(const DiscreteValues& values, LogarithmicVector<T>& result) const {
    result.param() = param.row(values()).transpose();
  }

  void restrict_back(const DiscreteValues& values, LogarithmicVector<T>& result) const {
    result.param() = param.col(values());
  }

  // Reshaping
  //--------------------------------------------------------------------------

  void transpose(LogarithmicMatrix& result) const {
    result.param() = param.transpose();
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T acc, T val) {
      return acc + EntropyLogOp<T>()(val);
    });
  }

  template <typename Op>
  T transform_sum(const LogarithmicMatrix& other, Op op) const {
    assert(param.rows() == other.rows() && param.cols() == other.cols());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), std::plus<T>(), op);
  }

  T cross_entropy(const LogarithmicMatrix& other) const {
    return transform_sum(other, EntropyLogOp<T>());
  }

  T kl_divergence(const LogarithmicMatrix& other) const {
    return transform_sum(other, KldLogOp<T>());
  }

  T sum_difference(const LogarithmicMatrix& other) const {
    return (param - other.param()).abs().sum();
  }

  T max_difference(const LogarithmicMatrix& other) const {
    return (param - other.param()).abs().maxCoeff();
  }

#if 0
  // Sampling
  //--------------------------------------------------------------------------

  /**
   * Returns a categorical distribution represented by this expression.
   */
  BivariateCategoricalDistribution<T> distribution() const {
    return { param, log_tag() };
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
#endif

}; // Impl

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x)
  : Object(std::make_unique<Impl>(rows, cols)) {
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(const Shape& shape, Exp<T> x)
  : Object(std::make_unique<Impl>(shape)) {
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values)
  : Object(std::make_unique<Impl>(rows, cols)) {
  assert(values.size() == rows * cols);
  std::copy(values.begin(), values.end(), impl().param.data());
}

template <typename T>
size_t LogarithmicMatrix<T>::rows() const {
  return impl().param.rows();
}

template <typename T>
size_t LogarithmicMatrix<T>::cols() const {
  return impl().param.cols();
}

template <typename T>
size_t LogarithmicMatrix<T>::size() const {
  return impl().param.size();
}

template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& LogarithmicMatrix<T>::param() {
  if (!impl_) {
    impl_.reset(new Impl);
  }
  return impl().param;
}

template <typename T>
const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& LogarithmicMatrix<T>::param() const {
  return impl().param;
}

template <typename T>
T LogarithmicMatrix<T>::log(size_t row, size_t col) const {
  return impl().param(row, col);
}

template <typename T>
T LogarithmicMatrix<T>::log(const DiscreteValues& values) const {
  assert(values.size() == 2);
  return impl().param(values[0], values[1]);
}

template <typename T>
ProbabilityMatrix<T> LogarithmicMatrix<T>::probability() const {
  return impl().param.exp();
}

template <typename T>
LogarithmicTable<T> LogarithmicMatrix<T>::table() const {
  return {{rows(), cols()}, param().data()};
}

template <typename T>
const typename LogarithmicMatrix<T>::VTable LogarithmicMatrix<T>::vtable{
  &LogarithmicMatrix<T>::Impl::multiply,
  &LogarithmicMatrix<T>::Impl::multiply,
  &LogarithmicMatrix<T>::Impl::multiply_in,
  &LogarithmicMatrix<T>::Impl::multiply_in,
  &LogarithmicMatrix<T>::Impl::divide,
  &LogarithmicMatrix<T>::Impl::divide_inverse,
  &LogarithmicMatrix<T>::Impl::divide,
  &LogarithmicMatrix<T>::Impl::divide_in,
  &LogarithmicMatrix<T>::Impl::divide_in,
  &LogarithmicMatrix<T>::Impl::multiply_front,
  &LogarithmicMatrix<T>::Impl::multiply_back,
  &LogarithmicMatrix<T>::Impl::multiply_in_front,
  &LogarithmicMatrix<T>::Impl::multiply_in_back,
  &LogarithmicMatrix<T>::Impl::divide_front,
  &LogarithmicMatrix<T>::Impl::divide_back,
  &LogarithmicMatrix<T>::Impl::divide_in_front,
  &LogarithmicMatrix<T>::Impl::divide_in_back,
  &LogarithmicMatrix<T>::Impl::power,
  &LogarithmicMatrix<T>::Impl::weighted_update,
  &LogarithmicMatrix<T>::Impl::maximum,
  &LogarithmicMatrix<T>::Impl::minimum,
  &LogarithmicMatrix<T>::Impl::maximum_front,
  &LogarithmicMatrix<T>::Impl::maximum_back,
  &LogarithmicMatrix<T>::Impl::minimum_front,
  &LogarithmicMatrix<T>::Impl::minimum_back,
  &LogarithmicMatrix<T>::Impl::restrict_front,
  &LogarithmicMatrix<T>::Impl::restrict_back,
  &LogarithmicMatrix<T>::Impl::transpose,
  &LogarithmicMatrix<T>::Impl::entropy,
  &LogarithmicMatrix<T>::Impl::cross_entropy,
  &LogarithmicMatrix<T>::Impl::kl_divergence,
  &LogarithmicMatrix<T>::Impl::sum_difference,
  &LogarithmicMatrix<T>::Impl::max_difference,
};

} // namespace libgm
