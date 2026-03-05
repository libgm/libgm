#pragma once

#include "../logarithmic_matrix.hpp"

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, Exp<T> x)
  : param_(rows, cols) {
  param_.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(const Shape& shape, Exp<T> x) {
  assert(shape.size() == 2);
  param_.resize(shape[0], shape[1]);
  param_.fill(x.lv);
}

template <typename T>
LogarithmicMatrix<T>::LogarithmicMatrix(size_t rows, size_t cols, std::initializer_list<T> values)
  : param_(rows, cols) {
  assert(values.size() == rows * cols);
  std::copy(values.begin(), values.end(), param_.data());
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::operator*(const Exp<T>& x) const {
  return {param_ + x.lv};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::operator*(const LogarithmicMatrix& other) const {
  return {param_ + other.param_};
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::operator*=(const Exp<T>& x) {
  param_.array() += x.lv;
  return *this;
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::operator*=(const LogarithmicMatrix& other) {
  param_ += other.param_;
  return *this;
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::operator/(const Exp<T>& x) const {
  return {param_ - x.lv};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::divide_inverse(const Exp<T>& x) const {
  return {x.lv - param_};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::operator/(const LogarithmicMatrix& other) const {
  return {param_ - other.param_};
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::operator/=(const Exp<T>& x) {
  param_.array() -= x.lv;
  return *this;
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::operator/=(const LogarithmicMatrix& other) {
  param_ -= other.param_;
  return *this;
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::multiply_front(const LogarithmicVector<T>& other) const {
  return {param_.colwise() + other.param()};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::multiply_back(const LogarithmicVector<T>& other) const {
  return {param_.rowwise() + other.param().transpose()};
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::multiply_in_front(const LogarithmicVector<T>& other) {
  param_.colwise() += other.param();
  return *this;
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::multiply_in_back(const LogarithmicVector<T>& other) {
  param_.rowwise() += other.param().transpose();
  return *this;
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::divide_front(const LogarithmicVector<T>& other) const {
  return {param_.colwise() - other.param()};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::divide_back(const LogarithmicVector<T>& other) const {
  return {param_.rowwise() - other.param().transpose()};
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::divide_in_front(const LogarithmicVector<T>& other) {
  param_.colwise() -= other.param();
  return *this;
}

template <typename T>
LogarithmicMatrix<T>& LogarithmicMatrix<T>::divide_in_back(const LogarithmicVector<T>& other) {
  param_.rowwise() -= other.param().transpose();
  return *this;
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::pow(T x) const {
  return {param_ * x};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::weighted_update(const LogarithmicMatrix& other, T x) const {
  return {(1 - x) * param_ + x * other.param_};
}

template <typename T>
Exp<T> LogarithmicMatrix<T>::maximum(std::vector<size_t>* values) const {
  if (values) {
    values->resize(2);
    size_t* data = values->data();
    return Exp<T>(param_.maxCoeff(data, data + 1));
  } else {
    return Exp<T>(param_.maxCoeff());
  }
}

template <typename T>
Exp<T> LogarithmicMatrix<T>::minimum(std::vector<size_t>* values) const {
  if (values) {
    values->resize(2);
    size_t* data = values->data();
    return Exp<T>(param_.minCoeff(data, data + 1));
  } else {
    return Exp<T>(param_.minCoeff());
  }
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::maximum_front(unsigned n) const {
  assert(n == 1);
  return {param_.rowwise().maxCoeff()};
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::maximum_back(unsigned n) const {
  assert(n == 1);
  return {param_.colwise().maxCoeff().transpose()};
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::minimum_front(unsigned n) const {
  assert(n == 1);
  return {param_.rowwise().minCoeff()};
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::minimum_back(unsigned n) const {
  assert(n == 1);
  return {param_.colwise().minCoeff().transpose()};
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::restrict_front(const std::vector<size_t>& values) const {
  assert(values.size() == 1);
  return {param_.row(values[0]).transpose()};
}

template <typename T>
LogarithmicVector<T> LogarithmicMatrix<T>::restrict_back(const std::vector<size_t>& values) const {
  assert(values.size() == 1);
  return {param_.col(values[0])};
}

template <typename T>
LogarithmicMatrix<T> LogarithmicMatrix<T>::transpose() const {
  return {param_.transpose()};
}

template <typename T>
T LogarithmicMatrix<T>::entropy() const {
  const T* begin = param_.data();
  const T* end = begin + param_.size();
  return std::accumulate(begin, end, T(0), [](T acc, T val) {
    return acc + EntropyLogOp<T>()(val);
  });
}

template <typename T>
T LogarithmicMatrix<T>::cross_entropy(const LogarithmicMatrix& other) const {
  assert(param_.rows() == other.rows() && param_.cols() == other.cols());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), EntropyLogOp<T>());
}

template <typename T>
T LogarithmicMatrix<T>::kl_divergence(const LogarithmicMatrix& other) const {
  assert(param_.rows() == other.rows() && param_.cols() == other.cols());
  return std::inner_product(param_.data(), param_.data() + param_.size(), other.param_.data(),
                            T(0), std::plus<T>(), KldLogOp<T>());
}

template <typename T>
T LogarithmicMatrix<T>::sum_diff(const LogarithmicMatrix& other) const {
  return (param_ - other.param_).abs().sum();
}

template <typename T>
T LogarithmicMatrix<T>::max_diff(const LogarithmicMatrix& other) const {
  return (param_ - other.param_).abs().maxCoeff();
}

template <typename T>
ProbabilityMatrix<T> LogarithmicMatrix<T>::probability() const {
  return {param_.exp()};
}

template <typename T>
LogarithmicTable<T> LogarithmicMatrix<T>::table() const {
  return {{rows(), cols()}, param_.data()};
}

} // namespace libgm
