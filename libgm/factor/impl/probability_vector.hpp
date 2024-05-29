#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

template <typename T>
struct ProbabilityVector<T>::Impl {

  /// The parameters of the factor, i.e., a vector of log-probabilities.
  DenseVector<T> param_;

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const ProbabilityVector& other) const override {
    return param_ == other.impl().param_;
  }

  void print(std::ostream& out) const override {
    out << param_;
  }

  void save(oarchive& ar) const {
    ar << param_;
  }

  void load(iarchive& ar) {
    ar >> param_;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply_in(const T& x) {
    param_.array() *= x;
  }

  void divide_in(const T& x) {
    param_.array() /= x;
  }

  void multiply_in(const ProbabilityVector<T>& other) {
    param_.array() *= other.impl().param_.array();
  }

  void divide_in(const ProbabilityVector<T>& other) {
    param_.array() /= other.impl().param_.array();
  }

  void normalize() {
    *this /= marginal();
  }

  ProbabilityVector<T> multiply(T x) const {
    return param_ * x;
  }

  ProbabilityVector<T> divide(T x) const {
    return param_ / x;
  }

  ProbabilityVector<T> divide_inverse(T x) const {
    return x / param_;
  }

  ProbabilityVector<T> pow(T x) const {
    return param_.pow(x);
  }

  ProbabilityVector<T> add(const ProbabilityVector<T>& other) {
    return param_ + other.param();
  }

  ProbabilityVector<T> multiply(const ProbabilityVector<T>& other) const {
    return param_ * other.impl().param_;
  }

  ProbabilityVector<T> divide(const ProbabilityVector<T>& other) const {
    return param_ / other.impl().param_;
  }

  ProbabilityVector<T> weighted_udpate(const ProbabilityVector<T>& other, T x) const {
    return (1 - x) * param_ + x * other.impl().param_;
  }

  // Conversions
  //-----------------------------------------------

  logarithmicVector<T> logarithmic() const {
    return param_.log();
  }

  ProbabilityTable<T> table() const {
    ...;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param_.sum();
  }

  T maximum() const {
    return param_.maxCoeff();
  }

  T minimum() const {
    return param_.minCoeff();
  }

  T maximum(std::size_t& row) const {
    return param_.maxCoeffIndex(&row);
  }

  T minimum(std::size_t& row) const {
    return param_.minCoeffIndex(&row);
  }

  bool normalizable() const {
    return max() > T();
  }

  CategoricalDistribution<T> distribution() const {
    return param_;
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    auto plus_entropy = compose_right(std::plus<T>(), EntropyOp<T>());
    return std::accumulate(param_.data(), param_.data() + param_.size(), T(0), plus_entropy);
  }

  T cross_entropy(const ProbabilityVector<T>& other) const {
    return transform_accumulate(EntropyOp<T>(), std::plus<T>(), T(0), param_, other.param());
  }

  T kl_divergence(const ProbabilityVector<T>& other) const {
    return transform_accumulate(KldOp<T>(), std::plus<T>(), T(0), param_, other.param());
  }

  T sum_diff(const ProbabilityVector<T>& other) const {
    return abs(param_ - other.impl().param_).sum();
  }

  T max_diff(const ProbabilityVector<T>& other) const {
    return abs(param_ - other.impl().param_).maxCoeff();
  }
}; // struct Impl

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length) {
  reset(length);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length, T x) {
  reset(length);
  impl().param_.fill(x);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(const DenseVector<T>& param)
  : Object(new Impl(param)) {}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(DenseVector<T>&& param)
  : Object(new Impl(std::move(param))) { }

template <typename T>
ProbabilityVector<T>::ProbabilityVector(std::initializer_list<T> params) {
  reset(param.size();)
  std::copy(params.begin(), params.end(), impl().param_.data());
}

template <typename T>
void ProbabilityVector<T>::reset(size_t length) {
  if (impl_) {
    impl().param.resize(length);
  } else {
    impl_.reset(new Impl(length));
  }
}

template <typename T>
size_t ProbabilityVector<T>::size() const {
  return impl().param_.size();
}

template <typename T>
T* ProbabilityVector<T>::begin() {
  return impl().param_.data();
}

template <typename T>
const T* ProbabilityVector<T>::begin() const {
  return impl().param_.data();
}

template <typename T>
T* ProbabilityVector<T>::end() {
  return impl().param_.data() + impl().param_.size();
}

template <typename T>
const T* ProbabilityVector<T>::end() const {
  return impl().param_.data() + impl().param_.size();
}

template <typename T>
DenseVector<T>& ProbabilityVector<T>::param() {
  return impl().param_;
}

template <typename T>
const DenseVector<T>& ProbabilityVector<T>::param() const {
  return impl().param_;
}

template <typename T>
T ProbabilityVector<T>::operator()(size_t row) const {
  return impl().param_[row];
}

template <typename T>
T ProbabilityVector<T>::operator()(const Assignment& a) const {
  return impl().param_[a.get<size_t>(0)];
}

template <typename T>
T ProbabilityVector<T>::log(size_t row) const {
  return std::log(impl().param_[row]);
}

template <typename T>
T ProbabilityVector<T>::log(const Assignment& a) const {
  return std::log(impl().param_[a.get<size_t>(0)]);
}

} // namespace libgm

#endif
