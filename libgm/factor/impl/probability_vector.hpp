#pragma once

#include "../probability_vector.hpp"

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct ProbabilityVector<T>::Impl : Object::Impl {

  /// The parameters of the factor, i.e., a vector of log-probabilities.
  Eigen::Array<T, Eigen::Dynamic, 1> param;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
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

  void multiply(const T& x, ProbabilityVector& result) const override {
    result.param() = a.param() * x;
  }

  void divide(const T& x, ProbabilityVector& result) const override {
    result.param() = param / x;
  }

  void divide_inverse(const T& x, ProbabilityVector& result) const {
    result.param() = x / param;
  }

  void multiply(const ProbabilityVector& other, ProbabilityVector& result) const {
    result.param() = param * other.param();
  }

  void divide(const ProbabilityVector& other, ProbabilityVector& result) const {
    result.param() = param / other.param();
  }

  void multiply_in(const T& x) {
    param *= x;
  }

  void divide_in(const T& x) {
    param /= x;
  }

  void multiply_in(const ProbabilityVector& other) {
    param *= other.param();
  }

  void divide_in(const ProbabilityVector& other) {
    param /= other.param();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  void power(T x, ProbabilityVector& result) const {
    result.param() = param.pow(x);
  }

  void weighted_update(const ProbabilityVector& other, T x, ProbabilityVector& result) const {
    result.param() = (1 - x) * param + x * other.param();
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param.sum();
  }

  T maximum(DiscreteValues* values) const {
    return values ? param.maxCoeff(values->resize(1)) : param.maxCoeff();
  }

  T minimum(DiscreteValues* values) const {
    return values ? param.minCoeff(values->resize(1)) : param.minCoeff();
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    param /= marginal();
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T acc, T val) {
      return acc + EntropyOp<T>()(val);
    });
  }

  template <typename Op>
  T transform_sum(const ProbabilityVector& other, Op op) const {
    assert(param.size() == other.size());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), std::plus<T>(), op);
  }

  T cross_entropy(const ProbabilityVector& other) const {
    return transform_sum(other, EntropyOp<T>());
  }

  T kl_divergence(const ProbabilityVector& other) const {
    return transform_sum(other, KldOp<T>());
  }

  T sum_difference(const ProbabilityVector& other) const {
    return (param - other.param()).abs().sum();
  }

  T max_difference(const ProbabilityVector& other) const {
    return (param - other.param()).abs().maxCoeff();
  }

}; // struct Impl

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length, T x)
  : Object(std::make_unique<Impl>(length)) {
  impl().param.fill(x);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(std::initializer_list<T> values)
  : Object(std::make_unique<Impl>(values.size())) {
  std::copy(values.begin(), values.end(), impl().param.data());
}

template <typename T>
size_t ProbabilityVector<T>::size() const {
  return impl().param.size();
}

template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>& ProbabilityVector<T>::param() {
  if (!impl_) {
    impl_.reset(new Impl);
  }
  return impl().param;
}

template <typename T>
const Eigen::Array<T, Eigen::Dynamic, 1>& ProbabilityVector<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbabilityVector<T>::operator()(size_t row) const {
  return impl().param[row];
}

template <typename T>
T ProbabilityVector<T>::operator()(const DiscreteValues& values) const {
  return impl().param[values()];
}

template <typename T>
T ProbabilityVector<T>::log(size_t row) const {
  return std::log(impl().param[row]);
}

template <typename T>
T ProbabilityVector<T>::log(const DiscreteValues& values) const {
  return std::log(impl().param[values()]);
}

template <typename T>
LogarithmicVector<T> ProbabilityVector<T>::logarithmic() const {
  return param().log();
}

template <typename T>
ProbabilityTable<T> ProbabilityVector<T>::table() const {
  return {{size()}, param().data()};
}

template <typename T>
const typename ProbabilityVector<T>::VTable ProbabilityVector<T>::vtable{
  &ProbabilityVector<T>::Impl::multiply,
  &ProbabilityVector<T>::Impl::multiply,
  &ProbabilityVector<T>::Impl::multiply_in,
  &ProbabilityVector<T>::Impl::multiply_in,
  &ProbabilityVector<T>::Impl::divide,
  &ProbabilityVector<T>::Impl::divide_inverse,
  &ProbabilityVector<T>::Impl::divide,
  &ProbabilityVector<T>::Impl::divide_in,
  &ProbabilityVector<T>::Impl::divide_in,
  &ProbabilityVector<T>::Impl::power,
  &ProbabilityVector<T>::Impl::weighted_update,
  &ProbabilityVector<T>::Impl::marginal,
  &ProbabilityVector<T>::Impl::maximum,
  &ProbabilityVector<T>::Impl::minimum,
  &ProbabilityVector<T>::Impl::normalize,
  &ProbabilityVector<T>::Impl::entropy,
  &ProbabilityVector<T>::Impl::cross_entropy,
  &ProbabilityVector<T>::Impl::kl_divergence,
  &ProbabilityVector<T>::Impl::sum_difference,
  &ProbabilityVector<T>::Impl::max_difference,
};

} // namespace libgm
