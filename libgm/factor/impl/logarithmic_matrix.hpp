#pragma once

#include "../logarithmic_matrix.hpp"

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/impl/logarithmic_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/serialization/eigen.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicMatrix<T>::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  DenseMatrix<T> param;

  // Constructors
  //--------------------------------------------------------------------------

  explicit Impl(size_t rows, size_t cols)
    : param(rows, cols) {}

  explicit Impl(DenseVector<T> param)
    : param(std::move(param)) {}

  // Utility functions
  //--------------------------------------------------------------------------

  const T* begin() {
    return param.data();
  }

  const T* end() {
    return param.data() + param.size();
  }

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const Object& other) override {
    return param == impl(other).param;
  }

  void print(std::ostream& out) const override {
    out << param;
  }

  void save(oarchive& ar) const override {
    ar << param;
  }

  void load(iarchive& ar) override {
    ar >> param;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(Exp<T> x) const {
    return std::make_unique<Impl>(param + x.lv);
  }

  ImplPtr divide(Exp<T> x) const {
    return std::make_unique<Impl>(param - x.lv);
  }

  ImplPtr divide_inverse(Exp<T> x) const {
    return std::make_unique<Impl>(x.lv - param);
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(param + impl(other).param);
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(param - impl(other).param);
  }

  void multiply_in(const Exp<T>& x) {
    param.array() += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    param.array() -= x.lv;
  }

  void multiply_in(const Object& other) {
    param += impl(other).param;
  }

  void divide_in(const Object& other){
    param -= impl(other).param;
  }

  // Join operations
  //--------------------------------------------------------------------------

  ImplPtr multiply_front(const Object& other) const {
    return std::make_unique<Impl>(param.colwise() + vec(other));
  }

  ImplPtr multiply_back(const Object& other) const {
    return std::make_unique<Impl>(param.rowwise() + vec(other).transpose());
  }

  ImplPtr divide_front(const Object& other) const {
    return std::make_unique<Impl>(param.colwise() - vec(other));
  }

  ImplPtr divide_back(const Object& other) const {
    return std::make_unique<Impl>(param.rowwise() - vec(other).transpose());
  }

  void multiply_in_front(const Object& other) {
    param.colwise() *= vec(other);
  }

  void multiply_in_back(const Object& other) {
    param.rowwise() *= vec(other).transpose();
  }

  void divide_in_front(const Object& other) {
    param.colwise() /= vec(other);
  }

  void divide_in_back(const Object& other) {
    param.rowwise() /= vec(other).transpose();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return std::make_unique<Impl>(param * x);
  }

  ImplPtr weighted_update(const Object& other, T x) const {
    return std::make_unique<Impl>(param * (1 - x) + impl(other).param * x);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    return Exp<T>(log_sum_exp...);
  }

  Exp<T> maximum(Values* values) const {
    if (values) {
      size_t* data = values->resize<size_t>(2);
      return Exp<T>(param.maxCoeff(data[0], data[1]));
    } else {
      return Exp<T>(param.maxCoeff());
    }
  }

  Exp<T> minimum(Values* values) const {
    if (values) {
      size_t* data = values->resize<size_t>(2);
      return Exp<T>(param.minCoeff(data[0], data[1]));
    } else {
      return Exp<T>(param.minCoeff());
    }
  }

  ImplPtr marginal_front(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.rowwise().logSumExp());
  }

  ImplPtr marginal_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.colwise().logSumExp());
  }

  ImplPtr maximum_front(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.rowwise().maxCoeff());
  }

  ImplPtr maximum_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.colwise().maxCoeff());
  }

  ImplPtr minimum_front(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.rowwise().minCoeff());
  }

  ImplPtr minimum_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<LogarithmicVector<T>::Impl>(param.colwise().minCoeff());
  }

  // Normalization
  //--------------------------------------------------------------------------
  void normalize() {
    param -= marginal().lv;
  }

  void normalize(unsigned nhead) const {
    assert(nhead == 1);
    Eigen::RowVector<T> vec = param.colwise().logSumExp();
    param.rowwise() -= vec;
  }

  // Restrictions
  //--------------------------------------------------------------------------

  ImplPtr restrict_head(const Values& values) const {
    return std::make_unique<LogarithmicVector<T>::Impl>(param.row(values.get<size_t>()));
  }

  LogarithmicVector<T> restrict_tail(const Values& values) const {
    return std::make_unique<LogarithmicVector<T>::Impl>(param.col(values.get<size_t>()));
  }

  // Reshaping
  //--------------------------------------------------------------------------

  ImplPtr transpose() const {
    return std::make_unique<Impl>(param.transpose());
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return *std::transform(begin(), end(), AccumulatingIterator<T>(), EntropyLogOp<T>());
  }

  template <typename Op>
  T transform_sum(const Object& other, Op op) const {
    const Impl& x = *this;
    const Impl& y = impl(other);
    return *std::transform(x.begin(), x.end(), y.begin(), AccumulatingIterator<T>(), op);
  }

  T cross_entropy(const Object& other) const {
    return transform_sum(other, EntropyLogOp<T>());
  }

  T kl_divergence(const Object& other) const {
    return transform_sum(other, KldLogOp<T>());
  }

  T sum_diff(const Object& other) const {
    return (param - impl(other).param).abs().sum();
  }

  T max_diff(const Object& other) const {
    return (param - impl(other).param).abs().maxCoeff();
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
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols) {
  reset(rows, cols);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(const Shape& shape) {
  assert(shape.size() == 2);
  reset(shape[0], shape[1]);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x) {
  reset(rows, cols);
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(const Shape& shape, Exp<T> x)
  : LogarithmicMatrix(shape) {
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(DenseMatrix<T> param)
  : Implements(new Impl(std::move(param))) {}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values) {
  assert(values.size() == rows * cols);
  reset(rows, cols);
  std::copy(values.begin(), values.end(), impl().param.data());
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
DenseMatrix<T>& LogarithmicMatrix<T>::param() {
  return impl().param;
}

template <typename T>
const DenseMatrix<T>& LogarithmicMatrix<T>::param() const {
  return impl().param;
}

template <typename T>
T LogarithmicMatrix<T>::log(size_t row, size_t col) const {
  return impl().param(row, col);
}

template <typename T>
T LogarithmicMatrix<T>::log(const Assignment& a) const {
  return impl().param(a.get<size_t>(0), a.get<size_t>(1));
}

ProbabilityMatrix<T> LogarithmicMatrix<T>::probability() const {
  return exp(impl().param);
}

LogarithmicTable<T> LogarithmicMatrix<T>::table() const {
  return table_from_matrix<log_tag>(derived()); // in table_function.hpp
}

}; // class LogarithmicMatrix

} // namespace libgm
