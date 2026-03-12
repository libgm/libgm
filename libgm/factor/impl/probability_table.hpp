#pragma once

#include "../probability_table.hpp"

#include <libgm/datastructure/table_operations.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace libgm {

template <typename T>
ProbabilityTable<T>::ProbabilityTable(T value)
  : param_(Shape()) {
  param_[0] = value;
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, T value)
  : param_(std::move(shape)) {
  param_.fill(value);
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, std::initializer_list<T> values)
  : param_(std::move(shape)) {
  assert(values.size() == size());
  std::copy(values.begin(), values.end(), param_.begin());
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, const T* values)
  : param_(std::move(shape)) {
  std::copy(values, values + size(), param_.begin());
}

template <typename T>
void ProbabilityTable<T>::reset(Shape shape) {
  param_.reset(std::move(shape));
  param_.fill(T(1));
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator*(T x) const {
  ProbabilityTable result;
  transform(param_, MultipliedBy<T>(x), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator*(const ProbabilityTable& other) const {
  ProbabilityTable result;
  transform(param_, other.param_, std::multiplies<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator*=(T x) {
  transform_in(param_, MultipliedBy<T>(x));
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator*=(const ProbabilityTable& other) {
  transform_in(param_, other.param_, std::multiplies<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator/(T x) const {
  ProbabilityTable result;
  transform(param_, DividedBy<T>(x), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_inverse(T x) const {
  ProbabilityTable result;
  transform(param_, Dividing<T>(x), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator/(const ProbabilityTable& other) const {
  ProbabilityTable result;
  transform(param_, other.param_, std::divides<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator/=(T x) {
  transform_in(param_, DividedBy<T>(x));
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator/=(const ProbabilityTable& other) {
  transform_in(param_, other.param_, std::divides<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply_front(const ProbabilityTable& other) const {
  ProbabilityTable result;
  join_front(param_, other.param_, std::multiplies<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply_back(const ProbabilityTable& other) const {
  ProbabilityTable result;
  join_back(param_, other.param_, std::multiplies<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply(const ProbabilityTable& other, const Dims& i, const Dims& j) const {
  ProbabilityTable result;
  join(param_, other.param_, i, j, std::multiplies<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in_front(const ProbabilityTable& other) {
  join_in_front(param_, other.param_, std::multiplies<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in_back(const ProbabilityTable& other) {
  join_in_back(param_, other.param_, std::multiplies<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in(const ProbabilityTable& other, const Dims& dims) {
  join_in(param_, other.param_, dims, std::multiplies<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_front(const ProbabilityTable& other) const {
  ProbabilityTable result;
  join_front(param_, other.param_, std::divides<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_back(const ProbabilityTable& other) const {
  ProbabilityTable result;
  join_back(param_, other.param_, std::divides<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide(const ProbabilityTable& other, const Dims& i, const Dims& j) const {
  ProbabilityTable result;
  join(param_, other.param_, i, j, std::divides<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in_front(const ProbabilityTable& other) {
  join_in_front(param_, other.param_, std::divides<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in_back(const ProbabilityTable& other) {
  join_in_back(param_, other.param_, std::divides<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in(const ProbabilityTable& other, const Dims& dims) {
  join_in(param_, other.param_, dims, std::divides<T>());
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::pow(T x) const {
  ProbabilityTable result;
  transform(param_, PowerOp<T>(x), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::weighted_update(const ProbabilityTable& other, T x) const {
  ProbabilityTable result;
  transform(param_, other.param_, WeightedPlus<T>(1 - x, x), result.param_);
  return result;
}

template <typename T>
T ProbabilityTable<T>::marginal() const {
  return std::accumulate(param_.begin(), param_.end(), T(0), std::plus<T>());
}

template <typename T>
T ProbabilityTable<T>::maximum(std::vector<size_t>* values) const {
  assert(param_.size() > 0);
  auto it = std::max_element(param_.begin(), param_.end());
  if (values) {
    *values = param_.shape().index(it - param_.begin());
  }
  return *it;
}

template <typename T>
T ProbabilityTable<T>::minimum(std::vector<size_t>* values) const {
  assert(param_.size() > 0);
  auto it = std::min_element(param_.begin(), param_.end());
  if (values) {
    *values = param_.shape().index(it - param_.begin());
  }
  return *it;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_front(unsigned n) const {
  ProbabilityTable result;
  aggregate_front(param_, n, T(0), std::plus<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_back(unsigned n) const {
  ProbabilityTable result;
  aggregate_back(param_, n, T(0), std::plus<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_dims(const Dims& retain) const {
  ProbabilityTable result;
  aggregate(param_, retain, T(0), std::plus<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_front(unsigned n) const {
  ProbabilityTable result;
  aggregate_front(param_, n, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_back(unsigned n) const {
  ProbabilityTable result;
  aggregate_back(param_, n, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_dims(const Dims& retain) const {
  ProbabilityTable result;
  aggregate(param_, retain, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_front(unsigned n) const {
  ProbabilityTable result;
  aggregate_front(param_, n, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_back(unsigned n) const {
  ProbabilityTable result;
  aggregate_back(param_, n, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_dims(const Dims& retain) const {
  ProbabilityTable result;
  aggregate(param_, retain, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
void ProbabilityTable<T>::normalize() {
  (*this) /= marginal();
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_front(const std::vector<size_t>& values) const {
  ProbabilityTable result;
  libgm::restrict_front(param_, values, result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_back(const std::vector<size_t>& values) const {
  ProbabilityTable result;
  libgm::restrict_back(param_, values, result.param_);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_dims(const Dims& dims, const std::vector<size_t>& values) const {
  ProbabilityTable result;
  libgm::restrict(param_, dims, values, result.param_);
  return result;
}

template <typename T>
T ProbabilityTable<T>::entropy() const {
  return std::accumulate(param_.begin(), param_.end(), T(0), [](T accu, T val) {
    return accu + EntropyOp<T>()(val);
  });
}

template <typename T>
T ProbabilityTable<T>::cross_entropy(const ProbabilityTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), EntropyOp<T>());
}

template <typename T>
T ProbabilityTable<T>::kl_divergence(const ProbabilityTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), KldOp<T>());
}

template <typename T>
T ProbabilityTable<T>::sum_diff(const ProbabilityTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), AbsDifference<T>());
}

template <typename T>
T ProbabilityTable<T>::max_diff(const ProbabilityTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            MaximumOp<T>(), AbsDifference<T>());
}

template <typename T>
LogarithmicTable<T> ProbabilityTable<T>::logarithmic() const {
  Table<T> result;
  transform(param_, LogarithmOp<T>(), result);
  return LogarithmicTable<T>(std::move(result));
}

template <typename T>
ProbabilityVector<T> ProbabilityTable<T>::vector() const {
  assert(arity() == 1);
  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(param_.data(), size());
}

template <typename T>
ProbabilityMatrix<T> ProbabilityTable<T>::matrix() const {
  assert(arity() == 2);
  using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  return Eigen::Map<const Array>(param_.data(), param_.size(0), param_.size(1));
}

} // namespace libgm
