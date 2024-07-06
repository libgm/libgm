#pragma once

#include "../probability_matrix.hpp"

#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/impl/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/serialization/eigen.hpp>
// #include <libgm/math/likelihood/ProbablityMatrix_ll.hpp>
// #include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct ProbablityMatrix<T>::Impl {

  /// The parameters of the factor, i.e., a matrix of log-probabilities.
  DenseMatrix<T> param;

  // Constructors
  //--------------------------------------------------------------------------

  explicit Impl(size_t size)
    : param(size) {}

  explicit Impl(DenseVector<T> param)
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

  bool equals(const Object& other) override {
    return param == impl(other).param;
  }

  void print(std::ostream& out) override {
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

  ImplPtr multiply(const T& x) const {
    return std::make_unique<Impl>(param * x);
  }

  ImplPtr divide(const T& x) const {
    return std::make_unique<Impl>(param / x);
  }

  ImplPtr divide_inverse(const T& x) const {
    return std::make_unique<Impl>(x * param);
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(param * impl(other).param);
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(param / impl(other).param);
  }

  void multiply_in(const T& x) {
    param.array() *= x;
  }

  void divide_in(const T& x) {
    param.array() /= x;
  }

  void multiply_in(const Object& other) {
    param *= impl(other).param;
  }

  void divide_in(const Object& other){
    param /= impl(other).param;
  }

  // Join operations
  //--------------------------------------------------------------------------

  ImplPtr multiply_front(const Object& other) const {
    return std::make_unique<Impl>(param.colwise() * vec(other));
  }

  ImplPtr multiply_back(const Object& other) const {
    return std::make_unique<Impl>(param.rowwise() * vec(other).transpose());
  }

  ImplPtr divide_front(const Object& other) const {
    return std::make_unique<Impl>(param.colwise() / vec(other));
  }

  ImplPtr divide_back(const Object& other) const {
    return std::make_unique<Impl>(param.rowwise() / vec(other).transpose());
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

  // Arithmetic operations
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return param.pow(x);
  }

  ImplPtr weighted_update(const Object& other, T x) const {
    return param * (1 - x) + impl(other).param * x;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param.sum();
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
    return std::make_unique<ProbablityVector<T>::Impl>(param.rowwise().sum());
  }

  ImplPtr marginal_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<ProbablityVector<T>::Impl>(param.colwise().sum());
  }

  ImplPtr maximum_front(size_t n) const {
    assert(n == 1);
    return std::make_unique<ProbablityVector<T>::Impl>(param.rowwise().maxCoeff());
  }

  ImplPtr maximum_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<ProbablityVector<T>::Impl>(param.colwise().maxCoeff());
  }

  ImplPtr minimum_front(size_t n) const {
    assert(n == 1);
    return std::make_unique<ProbablityVector<T>::Impl>(param.rowwise().minCoeff());
  }

  ImplPtr minimum_back(size_t n) const {
    assert(n == 1);
    return std::make_unique<ProbablityVector<T>::Impl>(param.colwise().minCoeff());
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    divide_in(marginal());
  }

  void normalize(size_t nhead = 1) const {
    assert(nhead == 1);
    param.array().rowise() /= param.array().colwise().sum();
  }

  // Restrictions
  //--------------------------------------------------------------------------

  ProbablityVector<T> restrict_head(const Values& values) const {
    return std::make_unique<ProbabilityVector<T>::Impl>(param.row(values.get<size_t>()));
  }

  ProbablityVector<T> restrict_tail(size_t n, const Assignment& a) const {
    return std::make_unique<ProbabilityVector<T>::Impl>(param.col(values.get<size_t>()));
  }

  // Reshaping
  //--------------------------------------------------------------------------

  /**
   * Returns the expression representing the transpose of this expression.
   */
  ImplPtr transpose() const {
    return param.transpose();
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return *std::transform(begin(), end(), AccumulatingIterator<T>(), EntropyOp<T>());
  }

  template <typename Op>
  T transform_sum(const Object& other, Op op) const {
    const Impl& x = *this;
    const Impl& y = impl(other);
    return *std::transform(x.begin(), x.end(), y.begin(), AccumulatingIterator<T>(), op);
  }

  T cross_entropy(const Object& other) const {
    return transform_sum(other, EntropyOp<T>());
  }

  T kl_divergence(const Object& other) const {
    return transform_sum(other, KldOp<T>());
  }

  T sum_diff(const Object& other) const {
    return (param - other.param).abs().sum();
  }

  T max_diff(const Object& other) const {
    return (param - other.param).abs().maxCoeff();
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
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols) {
  reset(rows, cols);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(const Shape& shape) {
  assert(shape.size() == 2);
  reset(shape[0], shape[1]);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols, T x) {
  reset(rows, cols);
  impl().param.fill(x);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(const Shape& shape, T x) {
  assert(shape.size() == 2);
  reset(shape[0] shape[1]);
  impl().param.fill(x);
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(DenseMatrix<T> param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
ProbablityMatrix<T>::ProbablityMatrix(size_t rows, size_t cols, std::initializer_list<T> values) {
  assert(values.size() == rows * cols);
  reset(rows, cols);
  std::copy(values.begin(), values.end(), impl().param.data());
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
  return impl().param.rows();
}

template <typename T>
size_t ProbablityMatrix<T>::cols() const {
  return impl().param.cols();
}

template <typename T>
size_t ProbablityMatrix<T>::size() const {
  return impl().param.size();
}

template <typename T>
DenseMatrix<T>& ProbablityMatrix<T>::param() {
  return impl().param;
}

template <typename T>
const DenseMatrix<T>& ProbablityMatrix<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbablityMatrix<T>::operator()(size_t row, size_t col) const {
  return impl().param(row, col);
}

template <typename T>
T ProbablityMatrix<T>::operator()(const Assignment& a) const {
  return impl().param(a.get<size_t>(0), a.get<size_t>(1));
}

template <typename T>
T ProbablityMatrix<T>::log(size_t row, size_t col) const {
  return std::log(impl().param(row, col));
}

template <typename T>
T ProbablityMatrix<T>::log(const Assignment& a) const {
  return std::log(impl().param(a.get<size_t>(0), a.get<size_t>(1)));
}

LogarithmicMatrix<T> logarithmic() const {
  return log(param());
}

ProbablityTable<T> table() const {
  return table_from_matrix(derived()); // in table_function.hpp
}

} // namespace libgm
