#pragma once

#include "../logarithmic_vector.hpp"

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>
// #include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
// #include <libgm/math/random/categorical_distribution.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicVector<T>::Impl : Object::Impl {

  /// The parameters of the factor, i.e., a vector of log-probabilities.
  Eigen::Array<T, Eigen::Dynamic, 1> param;

  template <typename ARCHIVE>
  void seralize(ARCHIVE& ar) {
    ar(param);
  }

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  explicit Impl(size_t size)
    : param(size) {}

  explicit Impl(const Shape& shape) {
    assert(shape.size() == 1);
    param.resize(shape[0]);
  }

  explicit Impl(Eigen::Array<T, Eigen::Dynamic, 1> param)
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

  void multiply(const Exp<T>& x, LogarithmicVector& result) const {
    result.param() = param + x.lv;
  }

  void divide(const Exp<T>& x, LogarithmicVector& result) const {
    result.param() = param - x.lv;
  }

  void divide_inverse(const Exp<T>& x, LogarithmicVector& result) const {
    result.param() = x.lv - param;
  }

  void multiply(const LogarithmicVector& other, LogarithmicVector& result) const {
    result.param() = param + other.param();
  }

  void divide(const LogarithmicVector& other, LogarithmicVector& result) const {
    result.param() = param - other.param();
  }

  void multiply_in(const Exp<T>& x) {
    param += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    param -= x.lv;
  }

  void multiply_in(const LogarithmicVector& other) {
    param += other.param();
  }

  void divide_in(const LogarithmicVector& other) {
    param -= other.param();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  void power(T x, LogarithmicVector& result) const {
    result.param() = param * x;
  }

  void weighted_update(const LogarithmicVector& other, T x, LogarithmicVector& result) const {
    result.param() = (1 - x) * param + x * other.param();
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(DiscreteValues* values) const {
    return Exp<T>(values ? param.maxCoeff(values->resize(1)) : param.maxCoeff());
  }

  Exp<T> minimum(DiscreteValues* values) const {
    return Exp<T>(values ? param.minCoeff(values->resize(1)) : param.minCoeff());
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T acc, T val) {
      return acc + EntropyLogOp<T>()(val);
    });
  }

  template <typename Op>
  T transform_sum(const LogarithmicVector& other, Op op) const {
    assert(param.size() == other.size());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), std::plus<T>(), op);
  }

  T cross_entropy(const LogarithmicVector& other) const {
    return transform_sum(other, EntropyLogOp<T>());
  }

  T kl_divergence(const LogarithmicVector& other) const {
    return transform_sum(other, KldLogOp<T>());
  }

  T sum_difference(const LogarithmicVector& other) const {
    return (param - other.param()).abs().sum();
  }

  T max_difference(const LogarithmicVector& other) const {
    return (param - other.param()).abs().maxCoeff();
  }

}; // struct Impl

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length, Exp<T> x)
  : Object(std::make_unique<Impl>(length)) {
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(const Shape& shape, Exp<T> x)
  : Object(std::make_unique<Impl>(shape)) {
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(std::initializer_list<T> list)
  : Object(std::make_unique<Impl>(list.size())) {
  std::copy(list.begin(), list.end(), impl().param.data());
}

template <typename T>
size_t LogarithmicVector<T>::size() const {
  return impl().param.size();
}

template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>& LogarithmicVector<T>::param() {
  if (!impl_) {
    impl_.reset(new Impl);
  }
  return impl().param;
}

template <typename T>
const Eigen::Array<T, Eigen::Dynamic, 1>& LogarithmicVector<T>::param() const {
  return impl().param;
}

template <typename T>
T LogarithmicVector<T>::log(size_t row) const {
  return impl().param[row];
}

template <typename T>
T LogarithmicVector<T>::log(const DiscreteValues& values) const {
  return impl().param[values()];
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator*(const Exp<T>& x) const {
  LogarithmicVector result;
  impl().multiply(x, result);
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator*(const LogarithmicVector& other) const {
  LogarithmicVector result;
  impl().multiply(other, result);
  return result;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator*=(const Exp<T>& x) {
  impl().multiply_in(x);
  return *this;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator*=(const LogarithmicVector& other) {
  impl().multiply_in(other);
  return *this;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator/(const Exp<T>& x) const {
  LogarithmicVector result;
  impl().divide(x, result);
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::divide_inverse(const Exp<T>& x) const {
  LogarithmicVector result;
  impl().divide_inverse(x, result);
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator/(const LogarithmicVector& other) const {
  LogarithmicVector result;
  impl().divide(other, result);
  return result;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator/=(const Exp<T>& x) {
  impl().divide_in(x);
  return *this;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator/=(const LogarithmicVector& other) {
  impl().divide_in(other);
  return *this;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::pow(T x) const {
  LogarithmicVector result;
  impl().power(x, result);
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::weighted_update(const LogarithmicVector& other, T x) const {
  LogarithmicVector result;
  impl().weighted_update(other, x, result);
  return result;
}

template <typename T>
Exp<T> LogarithmicVector<T>::maximum(DiscreteValues* values) const {
  return impl().maximum(values);
}

template <typename T>
Exp<T> LogarithmicVector<T>::minimum(DiscreteValues* values) const {
  return impl().minimum(values);
}

template <typename T>
T LogarithmicVector<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T LogarithmicVector<T>::cross_entropy(const LogarithmicVector& other) const {
  return impl().cross_entropy(other);
}

template <typename T>
T LogarithmicVector<T>::kl_divergence(const LogarithmicVector& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
T LogarithmicVector<T>::sum_diff(const LogarithmicVector& other) const {
  return impl().sum_difference(other);
}

template <typename T>
T LogarithmicVector<T>::max_diff(const LogarithmicVector& other) const {
  return impl().max_difference(other);
}

template <typename T>
ProbabilityVector<T> LogarithmicVector<T>::probability() const {
  return param().exp();
}

template <typename T>
LogarithmicTable<T> LogarithmicVector<T>::table() const {
  return {{size()}, param().data()};
}

template <typename T>
typename LogarithmicVector<T>::Impl& LogarithmicVector<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *static_cast<Impl*>(impl_.get());
}

template <typename T>
const typename LogarithmicVector<T>::Impl& LogarithmicVector<T>::impl() const {
  assert(impl_);
  return *static_cast<const Impl*>(impl_.get());
}

} // namespace libgm
