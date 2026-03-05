#pragma once

#include "../logarithmic_table.hpp"

#include <libgm/datastructure/table_operations.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Exp<T> value)
  : param_(Shape()) {
  param_[0] = value.lv;
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, Exp<T> value)
  : param_(std::move(shape)) {
  param_.fill(value.lv);
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, std::initializer_list<T> values)
  : param_(std::move(shape)) {
  assert(values.size() == size());
  std::copy(values.begin(), values.end(), param_.begin());
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, const T* values)
  : param_(std::move(shape)) {
  std::copy(values, values + size(), param_.begin());
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator*(const Exp<T>& x) const {
  LogarithmicTable result;
  transform(param_, IncrementedBy<T>(x.lv), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator*(const LogarithmicTable& other) const {
  LogarithmicTable result;
  transform(param_, other.param_, std::plus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator*=(const Exp<T>& x) {
  transform_in(param_, IncrementedBy<T>(x.lv));
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator*=(const LogarithmicTable& other) {
  transform_in(param_, other.param_, std::plus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator/(const Exp<T>& x) const {
  LogarithmicTable result;
  transform(param_, DecrementedBy<T>(x.lv), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_inverse(const Exp<T>& x) const {
  LogarithmicTable result;
  transform(param_, SubtractedFrom<T>(x.lv), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator/(const LogarithmicTable& other) const {
  LogarithmicTable result;
  transform(param_, other.param_, std::minus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator/=(const Exp<T>& x) {
  transform_in(param_, DecrementedBy<T>(x.lv));
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator/=(const LogarithmicTable& other) {
  transform_in(param_, other.param_, std::minus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply_front(const LogarithmicTable& other) const {
  LogarithmicTable result;
  join_front(param_, other.param_, std::plus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply_back(const LogarithmicTable& other) const {
  LogarithmicTable result;
  join_back(param_, other.param_, std::plus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply(const LogarithmicTable& other, const Dims& i, const Dims& j) const {
  LogarithmicTable result;
  join(param_, other.param_, i, j, std::plus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in_front(const LogarithmicTable& other) {
  join_in_front(param_, other.param_, std::plus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in_back(const LogarithmicTable& other) {
  join_in_back(param_, other.param_, std::plus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in(const LogarithmicTable& other, const Dims& dims) {
  join_in(param_, other.param_, dims, std::plus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_front(const LogarithmicTable& other) const {
  LogarithmicTable result;
  join_front(param_, other.param_, std::minus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_back(const LogarithmicTable& other) const {
  LogarithmicTable result;
  join_back(param_, other.param_, std::minus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide(const LogarithmicTable& other, const Dims& i, const Dims& j) const {
  LogarithmicTable result;
  join(param_, other.param_, i, j, std::minus<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in_front(const LogarithmicTable& other) {
  join_in_front(param_, other.param_, std::minus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in_back(const LogarithmicTable& other) {
  join_in_back(param_, other.param_, std::minus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in(const LogarithmicTable& other, const Dims& dims) {
  join_in(param_, other.param_, dims, std::minus<T>());
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::pow(T x) const {
  LogarithmicTable result;
  transform(param_, MultipliedBy<T>(x), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::weighted_update(const LogarithmicTable& other, T x) const {
  LogarithmicTable result;
  transform(param_, other.param_, WeightedPlus<T>(1 - x, x), result.param_);
  return result;
}

template <typename T>
Exp<T> LogarithmicTable<T>::maximum(std::vector<size_t>* values) const {
  assert(param_.size() > 0);
  auto it = std::max_element(param_.begin(), param_.end());
  if (values) {
    *values = param_.shape().index(it - param_.begin());
  }
  return Exp<T>(*it);
}

template <typename T>
Exp<T> LogarithmicTable<T>::minimum(std::vector<size_t>* values) const {
  assert(param_.size() > 0);
  auto it = std::min_element(param_.begin(), param_.end());
  if (values) {
    *values = param_.shape().index(it - param_.begin());
  }
  return Exp<T>(*it);
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_front(unsigned n) const {
  LogarithmicTable result;
  aggregate_front(param_, n, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_back(unsigned n) const {
  LogarithmicTable result;
  aggregate_back(param_, n, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_dims(const Dims& dims) const {
  LogarithmicTable result;
  aggregate(param_, dims, -inf<T>(), MaximumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_front(unsigned n) const {
  LogarithmicTable result;
  aggregate_front(param_, n, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_back(unsigned n) const {
  LogarithmicTable result;
  aggregate_back(param_, n, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_dims(const Dims& dims) const {
  LogarithmicTable result;
  aggregate(param_, dims, inf<T>(), MinimumOp<T>(), result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_front(const std::vector<size_t>& values) const {
  LogarithmicTable result;
  libgm::restrict_front(param_, values, result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_back(const std::vector<size_t>& values) const {
  LogarithmicTable result;
  libgm::restrict_back(param_, values, result.param_);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_dims(const Dims& dims, const std::vector<size_t>& values) const {
  LogarithmicTable result;
  libgm::restrict(param_, dims, values, result.param_);
  return result;
}

template <typename T>
T LogarithmicTable<T>::entropy() const {
  return std::accumulate(param_.begin(), param_.end(), T(0), [](T accu, T val) {
    return accu + EntropyLogOp<T>()(val);
  });
}

template <typename T>
T LogarithmicTable<T>::cross_entropy(const LogarithmicTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), EntropyLogOp<T>());
}

template <typename T>
T LogarithmicTable<T>::kl_divergence(const LogarithmicTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), KldLogOp<T>());
}

template <typename T>
T LogarithmicTable<T>::sum_diff(const LogarithmicTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            std::plus<T>(), AbsDifference<T>());
}

template <typename T>
T LogarithmicTable<T>::max_diff(const LogarithmicTable& other) const {
  assert(param_.shape() == other.shape());
  return std::inner_product(param_.begin(), param_.end(), other.param_.begin(), T(0),
                            MaximumOp<T>(), AbsDifference<T>());
}

template <typename T>
ProbabilityTable<T> LogarithmicTable<T>::probability() const {
  ProbabilityTable<T> result;
  transform(param_, ExponentOp<T>(), result.param());
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicTable<T>::vector() const {
  assert(arity() == 1);
  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(param_.data(), size());
}

template <typename T>
LogarithmicMatrix<T> LogarithmicTable<T>::matrix() const {
  assert(arity() == 2);
  using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  return Eigen::Map<const Array>(param_.data(), param_.size(0), param_.size(1));
}

} // namespace libgm
