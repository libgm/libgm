#pragma once

#include "../probability_vector.hpp"

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length, T x)
  : param_(length) {
  param_.fill(x);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(const Shape& shape, T x) {
  assert(shape.size() == 1);
  param_.resize(shape[0]);
  param_.fill(x);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(std::initializer_list<T> values)
  : param_(values.size()) {
  std::copy(values.begin(), values.end(), param_.data());
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::operator*(T x) const {
  return {param_ * x};
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::operator*(const ProbabilityVector& other) const {
  return {param_ * other.param_};
}

template <typename T>
ProbabilityVector<T>& ProbabilityVector<T>::operator*=(T x) {
  param_ *= x;
  return *this;
}

template <typename T>
ProbabilityVector<T>& ProbabilityVector<T>::operator*=(const ProbabilityVector& other) {
  param_ *= other.param_;
  return *this;
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::operator/(T x) const {
  return {param_ / x};
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::divide_inverse(T x) const {
  return {x / param_};
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::operator/(const ProbabilityVector& other) const {
  return {param_ / other.param_};
}

template <typename T>
ProbabilityVector<T>& ProbabilityVector<T>::operator/=(T x) {
  param_ /= x;
  return *this;
}

template <typename T>
ProbabilityVector<T>& ProbabilityVector<T>::operator/=(const ProbabilityVector& other) {
  param_ /= other.param_;
  return *this;
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::pow(T x) const {
  return {param_.pow(x)};
}

template <typename T>
ProbabilityVector<T> ProbabilityVector<T>::weighted_update(const ProbabilityVector& other, T x) const {
  return {(1 - x) * param_ + x * other.param_};
}

template <typename T>
T ProbabilityVector<T>::marginal() const {
  return param_.sum();
}

template <typename T>
T ProbabilityVector<T>::maximum(std::vector<size_t>* values) const {
  return values ? param_.maxCoeff((values->resize(1), values->data())) : param_.maxCoeff();
}

template <typename T>
T ProbabilityVector<T>::minimum(std::vector<size_t>* values) const {
  return values ? param_.minCoeff((values->resize(1), values->data())) : param_.minCoeff();
}

template <typename T>
void ProbabilityVector<T>::normalize() {
  param_ /= marginal();
}

template <typename T>
T ProbabilityVector<T>::entropy() const {
  const T* begin = param_.data();
  const T* end = begin + param_.size();
  return std::accumulate(begin, end, T(0), [](T acc, T val) {
    return acc + EntropyOp<T>()(val);
  });
}

template <typename T>
T ProbabilityVector<T>::cross_entropy(const ProbabilityVector& other) const {
  assert(param_.size() == other.size());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), EntropyOp<T>());
}

template <typename T>
T ProbabilityVector<T>::kl_divergence(const ProbabilityVector& other) const {
  assert(param_.size() == other.size());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), KldOp<T>());
}

template <typename T>
T ProbabilityVector<T>::sum_diff(const ProbabilityVector& other) const {
  return (param_ - other.param_).abs().sum();
}

template <typename T>
T ProbabilityVector<T>::max_diff(const ProbabilityVector& other) const {
  return (param_ - other.param_).abs().maxCoeff();
}

template <typename T>
LogarithmicVector<T> ProbabilityVector<T>::logarithmic() const {
  return param_.log();
}

template <typename T>
ProbabilityTable<T> ProbabilityVector<T>::table() const {
  return {{size()}, param_.data()};
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const ProbabilityVector<T>& f) {
  return out << "ProbabilityVector(" << f.param().transpose() << ")";
}

} // namespace libgm
