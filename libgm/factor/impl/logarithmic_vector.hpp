#pragma once

#include "../logarithmic_vector.hpp"

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length, Exp<T> x)
  : param_(length) {
  param_.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(const Shape& shape, Exp<T> x) {
  assert(shape.size() == 1);
  param_.resize(shape[0]);
  param_.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(std::initializer_list<T> list)
  : param_(list.size()) {
  std::copy(list.begin(), list.end(), param_.data());
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator*(const Exp<T>& x) const {
  return {param_ + x.lv};
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator*(const LogarithmicVector& other) const {
  return {param_ + other.param_};
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator*=(const Exp<T>& x) {
  param_ += x.lv;
  return *this;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator*=(const LogarithmicVector& other) {
  param_ += other.param_;
  return *this;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator/(const Exp<T>& x) const {
  return {param_ - x.lv};
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::divide_inverse(const Exp<T>& x) const {
  return {x.lv - param_};
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::operator/(const LogarithmicVector& other) const {
  return {param_ - other.param_};
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator/=(const Exp<T>& x) {
  param_ -= x.lv;
  return *this;
}

template <typename T>
LogarithmicVector<T>& LogarithmicVector<T>::operator/=(const LogarithmicVector& other) {
  param_ -= other.param_;
  return *this;
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::pow(T x) const {
  return {param_ * x};
}

template <typename T>
LogarithmicVector<T> LogarithmicVector<T>::weighted_update(const LogarithmicVector& other, T x) const {
  return {(1 - x) * param_ + x * other.param_};
}

template <typename T>
Exp<T> LogarithmicVector<T>::maximum(std::vector<size_t>* values) const {
  return Exp<T>(values ? param_.maxCoeff((values->resize(1), values->data())) : param_.maxCoeff());
}

template <typename T>
Exp<T> LogarithmicVector<T>::minimum(std::vector<size_t>* values) const {
  return Exp<T>(values ? param_.minCoeff((values->resize(1), values->data())) : param_.minCoeff());
}

template <typename T>
T LogarithmicVector<T>::entropy() const {
  const T* begin = param_.data();
  const T* end = begin + param_.size();
  return std::accumulate(begin, end, T(0), [](T acc, T val) {
    return acc + EntropyLogOp<T>()(val);
  });
}

template <typename T>
T LogarithmicVector<T>::cross_entropy(const LogarithmicVector& other) const {
  assert(param_.size() == other.size());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), EntropyLogOp<T>());
}

template <typename T>
T LogarithmicVector<T>::kl_divergence(const LogarithmicVector& other) const {
  assert(param_.size() == other.size());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), KldLogOp<T>());
}

template <typename T>
T LogarithmicVector<T>::sum_diff(const LogarithmicVector& other) const {
  return (param_ - other.param_).abs().sum();
}

template <typename T>
T LogarithmicVector<T>::max_diff(const LogarithmicVector& other) const {
  return (param_ - other.param_).abs().maxCoeff();
}

template <typename T>
ProbabilityVector<T> LogarithmicVector<T>::probability() const {
  return {param_.exp()};
}

template <typename T>
LogarithmicTable<T> LogarithmicVector<T>::table() const {
  return {{size()}, param_.data()};
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const LogarithmicVector<T>& f) {
  return out << "LogarithmicVector(" << f.param().transpose() << ")";
}

} // namespace libgm
