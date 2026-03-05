#pragma once

#include "../probability_matrix.hpp"

#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(size_t rows, size_t cols, T x)
  : param_(rows, cols) {
  param_.fill(x);
}

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(const Shape& shape, T x) {
  assert(shape.size() == 2);
  param_.resize(shape[0], shape[1]);
  param_.fill(x);
}

template <typename T>
ProbabilityMatrix<T>::ProbabilityMatrix(size_t rows, size_t cols, std::initializer_list<T> values)
  : param_(rows, cols) {
  assert(values.size() == rows * cols);
  std::copy(values.begin(), values.end(), param_.data());
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator*(T x) const {
  return {param_ * x};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator*(const ProbabilityMatrix& other) const {
  return {param_ * other.param_};
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator*=(T x) {
  param_ *= x;
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator*=(const ProbabilityMatrix& other) {
  param_ *= other.param_;
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator/(T x) const {
  return {param_ / x};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_inverse(T x) const {
  return {x / param_};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::operator/(const ProbabilityMatrix& other) const {
  return {param_ / other.param_};
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator/=(T x) {
  param_ /= x;
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::operator/=(const ProbabilityMatrix& other) {
  param_ /= other.param_;
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::multiply_front(const ProbabilityVector<T>& other) const {
  return {param_.colwise() * other.param()};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::multiply_back(const ProbabilityVector<T>& other) const {
  return {param_.rowwise() * other.param().transpose()};
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::multiply_in_front(const ProbabilityVector<T>& other) {
  param_.colwise() *= other.param();
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::multiply_in_back(const ProbabilityVector<T>& other) {
  param_.rowwise() *= other.param().transpose();
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_front(const ProbabilityVector<T>& other) const {
  return {param_.colwise() / other.param()};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::divide_back(const ProbabilityVector<T>& other) const {
  return {param_.rowwise() / other.param().transpose()};
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::divide_in_front(const ProbabilityVector<T>& other) {
  param_.colwise() /= other.param();
  return *this;
}

template <typename T>
ProbabilityMatrix<T>& ProbabilityMatrix<T>::divide_in_back(const ProbabilityVector<T>& other) {
  param_.rowwise() /= other.param().transpose();
  return *this;
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::pow(T x) const {
  return {param_.pow(x)};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::weighted_update(const ProbabilityMatrix& other, T x) const {
  return {(1 - x) * param_ + x * other.param_};
}

template <typename T>
T ProbabilityMatrix<T>::marginal() const {
  return param_.sum();
}

template <typename T>
T ProbabilityMatrix<T>::maximum(std::vector<size_t>* values) const {
  if (values) {
    values->resize(2);
    size_t* data = values->data();
    return param_.maxCoeff(data, data + 1);
  } else {
    return param_.maxCoeff();
  }
}

template <typename T>
T ProbabilityMatrix<T>::minimum(std::vector<size_t>* values) const {
  if (values) {
    values->resize(2);
    size_t* data = values->data();
    return param_.minCoeff(data, data + 1);
  } else {
    return param_.minCoeff();
  }
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::marginal_front(unsigned n) const {
  assert(n == 1);
  return {param_.rowwise().sum()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::marginal_back(unsigned n) const {
  assert(n == 1);
  return {param_.colwise().sum().transpose()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::maximum_front(unsigned n) const {
  assert(n == 1);
  return {param_.rowwise().maxCoeff()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::maximum_back(unsigned n) const {
  assert(n == 1);
  return {param_.colwise().maxCoeff().transpose()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::minimum_front(unsigned n) const {
  assert(n == 1);
  return {param_.rowwise().minCoeff()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::minimum_back(unsigned n) const {
  assert(n == 1);
  return {param_.colwise().minCoeff().transpose()};
}

template <typename T>
void ProbabilityMatrix<T>::normalize() {
  param_ /= marginal();
}

template <typename T>
void ProbabilityMatrix<T>::normalize_head(unsigned nhead) {
  assert(nhead == 1);
  param_.rowwise() /= param_.colwise().sum();
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::restrict_front(const std::vector<size_t>& values) const {
  assert(values.size() == 1);
  return {param_.row(values[0]).transpose()};
}

template <typename T>
ProbabilityVector<T> ProbabilityMatrix<T>::restrict_back(const std::vector<size_t>& values) const {
  assert(values.size() == 1);
  return {param_.col(values[0])};
}

template <typename T>
ProbabilityMatrix<T> ProbabilityMatrix<T>::transpose() const {
  return {param_.transpose()};
}

template <typename T>
T ProbabilityMatrix<T>::entropy() const {
  const T* begin = param_.data();
  const T* end = begin + param_.size();
  return std::accumulate(begin, end, T(0), [](T acc, T val) {
    return acc + EntropyOp<T>()(val);
  });
}

template <typename T>
T ProbabilityMatrix<T>::cross_entropy(const ProbabilityMatrix& other) const {
  assert(param_.rows() == other.rows() && param_.cols() == other.cols());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), EntropyOp<T>());
}

template <typename T>
T ProbabilityMatrix<T>::kl_divergence(const ProbabilityMatrix& other) const {
  assert(param_.rows() == other.rows() && param_.cols() == other.cols());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), KldOp<T>());
}

template <typename T>
T ProbabilityMatrix<T>::sum_diff(const ProbabilityMatrix& other) const {
  return (param_ - other.param_).abs().sum();
}

template <typename T>
T ProbabilityMatrix<T>::max_diff(const ProbabilityMatrix& other) const {
  return (param_ - other.param_).abs().maxCoeff();
}

template <typename T>
LogarithmicMatrix<T> ProbabilityMatrix<T>::logarithmic() const {
  return {param_.log()};
}

template <typename T>
ProbabilityTable<T> ProbabilityMatrix<T>::table() const {
  return {{rows(), cols()}, param_.data()};
}

} // namespace libgm
