#pragma once

#include "../logarithmic_table.hpp"

#include <libgm/datastructure/table_operations.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>
// #include <libgm/math/likelihood/logarithmic_table_ll.hpp>
// #include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include <iostream>

namespace libgm {

template <typename T>
struct LogarithmicTable<T>::Impl : Object::Impl {
  Table<T> param;

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(param);
  }

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  explicit Impl(Shape shape)
    : param(std::move(shape)) {}

  explicit Impl(Table<T> param)
    : param(std::move(param)) {}

  const T* begin() const {
    return param.begin();
  }

  const T* end() const {
    return param.end();
  }

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    out << param;
  }

  // Assignment
  //--------------------------------------------------------------------------

  void assign(const Exp<T>& x) {
    param.reset({});
    param[0] = x.lv;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply(const Exp<T>& x, LogarithmicTable& result) const {
    transform(param, IncrementedBy<T>(x.lv), result.param());
  }

  void divide(const Exp<T>& x, LogarithmicTable& result) const {
    transform(param, DecrementedBy<T>(x.lv), result.param());
  }

  void divide_inverse(const Exp<T>& x, LogarithmicTable& result) const {
    transform(param, SubtractedFrom<T>(x.lv), result.param());
  }

  void multiply(const LogarithmicTable& other, LogarithmicTable& result) const {
    transform(param, other.param(), std::plus<T>(), result.param());
  }

  void divide(const LogarithmicTable& other, LogarithmicTable& result) const {
    transform(param, other.param(), std::minus<T>(), result.param());
  }

  void multiply_in(const Exp<T>& x) {
    transform_in(param, IncrementedBy<T>(x.lv));
  }

  void divide_in(const Exp<T>& x) {
    transform_in(param, DecrementedBy<T>(x.lv));
  }

  void multiply_in(const LogarithmicTable& other) {
    transform_in(param, other.param(), std::plus<T>());
  }

  void divide_in(const LogarithmicTable& other) {
    transform_in(param, other.param(), std::minus<T>());
  }

  // Join operations
  //--------------------------------------------------------------------------

  void multiply_front(const LogarithmicTable& other, LogarithmicTable& result) const {
    join_front(param, other.param(), std::plus<T>(), result.param());
  }

  void multiply_back(const LogarithmicTable& other, LogarithmicTable& result) const {
    join_back(param, other.param(), std::plus<T>(), result.param());
  }

  void multiply_dims(const LogarithmicTable& other, const Dims& i, const Dims& j, LogarithmicTable& result) const {
    join(param, other.param(), i, j, std::plus<T>(), result.param());
  }

  void divide_front(const LogarithmicTable& other, LogarithmicTable& result) const {
    join_front(param, other.param(), std::minus<T>(), result.param());
  }

  void divide_back(const LogarithmicTable& other, LogarithmicTable& result) const {
    join_back(param, other.param(), std::minus<T>(), result.param());
  }

  void divide_dims(const LogarithmicTable& other, const Dims& i, const Dims& j, LogarithmicTable& result) const {
    join(param, other.param(), i, j, std::minus<T>(), result.param());
  }

  void multiply_in_front(const LogarithmicTable& other) {
    join_in_front(param, other.param(), std::plus<T>());
  }

  void multiply_in_back(const LogarithmicTable& other) {
    join_in_back(param, other.param(), std::plus<T>());
  }

  void multiply_in_dims(const LogarithmicTable& other, const Dims& dims) {
    join_in(param, other.param(), dims, std::plus<T>());
  }

  void divide_in_front(const LogarithmicTable& other) {
    join_in_front(param, other.param(), std::minus<T>());
  }

  void divide_in_back(const LogarithmicTable& other) {
    join_in_back(param, other.param(), std::minus<T>());
  }

  void divide_in_dims(const LogarithmicTable& other, const Dims& dims) {
    join_in(param, other.param(), dims, std::minus<T>());
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  void power(T x, LogarithmicTable& result) const {
    transform(param, MultipliedBy<T>(x), result.param());
  }

  void weighted_update(const LogarithmicTable& other, T x, LogarithmicTable& result) const {
    transform(param, other.param(), WeightedPlus<T>(1 - x, x), result.param());
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> maximum(DiscreteValues* values) const {
    auto it = std::max_element(begin(), end());
    if (values) {
      *values = param.shape().index(it - begin());
    }
    return Exp<T>(*it);
  }

  Exp<T> minimum(DiscreteValues* values) const {
    auto it = std::min_element(begin(), end());
    if (values) {
      *values = param.shape().index(it - begin());
    }
    return Exp<T>(*it);
  }

  void maximum_front(unsigned n, LogarithmicTable& result) const {
    aggregate_front(param, n, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void maximum_back(unsigned n, LogarithmicTable& result) const {
    aggregate_back(param, n, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void maximum_dims(const Dims& retain, LogarithmicTable& result) const {
    aggregate(param, retain, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void minimum_front(unsigned n, LogarithmicTable& result) const {
    aggregate_front(param, n, inf<T>(), MinimumOp<T>(), result.param());
  }

  void minimum_back(unsigned n, LogarithmicTable& result) const {
    aggregate_back(param, n, inf<T>(), MinimumOp<T>(), result.param());
  }

  void minimum_dims(const Dims& retain, LogarithmicTable& result) const {
    aggregate(param, retain, inf<T>(), MinimumOp<T>(), result.param());
  }

  // Restrictions
  //--------------------------------------------------------------------------

  void restrict_front(const DiscreteValues& values, LogarithmicTable& result) const {
    libgm::restrict_front(param, values.vec(), result.param());
  }

  void restrict_back(const DiscreteValues& values, LogarithmicTable& result) const {
    libgm::restrict_back(param, values.vec(), result.param());
  }

  void restrict_dims(const Dims& dims, const DiscreteValues& values, LogarithmicTable& result) const {
    libgm::restrict(param, dims, values.vec(), result.param());
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T accu, T val) {
      return accu + EntropyLogOp<T>()(val);
    });
  }

  template <typename AccuOp, typename TransOp>
  T transform_accumulate(const LogarithmicTable& other, AccuOp accu, TransOp trans) const {
    assert(param.shape() == other.shape());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), accu, trans);
  }

  T cross_entropy(const LogarithmicTable& other) const {
    return transform_accumulate(other, std::plus<T>(), EntropyLogOp<T>());
  }

  T kl_divergence(const LogarithmicTable& other) const {
    return transform_accumulate(param, std::plus<T>(), KldLogOp<T>());
  }

  T sum_difference(const LogarithmicTable& other) const {
    return transform_accumulate(param, std::plus<T>(), AbsDifference<T>());
  }

  T max_difference(const LogarithmicTable& other) const {
    return transform_accumulate(param, MaximumOp<T>(), AbsDifference<T>());
  }

}; // class LogarithmicTable<T>::Impl

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Exp<T> value)
  : Object(std::make_unique<Impl>()) {
  param()[0] = value.lv;
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, Exp<T> value)
  : Object(std::make_unique<Impl>(std::move(shape))) {
  param().fill(value.lv);
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, std::initializer_list<T> values)
  : Object(std::make_unique<Impl>(std::move(shape))) {
  assert(values.size() == size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Shape shape, const T* values)
  : Object(std::make_unique<Impl>(std::move(shape))) {
  std::copy(values, values + size(), param().begin());
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Table<T> param)
  : Object(std::make_unique<Impl>(std::move(param))) {}

template <typename T>
size_t LogarithmicTable<T>::arity() const {
  return param().arity();
}

template <typename T>
size_t LogarithmicTable<T>::size() const {
  return param().size();
}

template <typename T>
const Shape& LogarithmicTable<T>::shape() const {
  return param().shape();
}

template <typename T>
Table<T>& LogarithmicTable<T>::param() {
  if (!impl_) {
    impl_.reset(new Impl);
  }
  return impl().param;
}

template <typename T>
const Table<T>& LogarithmicTable<T>::param() const {
  return impl().param;
}

template <typename T>
Exp<T> LogarithmicTable<T>::operator()(const DiscreteValues& values) const {
  return Exp<T>(param()(values.vec()));
}

template <typename T>
T LogarithmicTable<T>::log(const DiscreteValues& values) const {
  return param()(values.vec());
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator*(const Exp<T>& x) const {
  LogarithmicTable result;
  impl().multiply(x, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator*(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().multiply(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator*=(const Exp<T>& x) {
  impl().multiply_in(x);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator*=(const LogarithmicTable& other) {
  impl().multiply_in(other);
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator/(const Exp<T>& x) const {
  LogarithmicTable result;
  impl().divide(x, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_inverse(const Exp<T>& x) const {
  LogarithmicTable result;
  impl().divide_inverse(x, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::operator/(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().divide(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator/=(const Exp<T>& x) {
  impl().divide_in(x);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::operator/=(const LogarithmicTable& other) {
  impl().divide_in(other);
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply_front(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().multiply_front(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply_back(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().multiply_back(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::multiply(const LogarithmicTable& other, const Dims& i, const Dims& j) const {
  LogarithmicTable result;
  impl().multiply_dims(other, i, j, result);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in_front(const LogarithmicTable& other) {
  impl().multiply_in_front(other);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in_back(const LogarithmicTable& other) {
  impl().multiply_in_back(other);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::multiply_in(const LogarithmicTable& other, const Dims& dims) {
  impl().multiply_in_dims(other, dims);
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_front(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().divide_front(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide_back(const LogarithmicTable& other) const {
  LogarithmicTable result;
  impl().divide_back(other, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::divide(const LogarithmicTable& other, const Dims& i, const Dims& j) const {
  LogarithmicTable result;
  impl().divide_dims(other, i, j, result);
  return result;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in_front(const LogarithmicTable& other) {
  impl().divide_in_front(other);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in_back(const LogarithmicTable& other) {
  impl().divide_in_back(other);
  return *this;
}

template <typename T>
LogarithmicTable<T>& LogarithmicTable<T>::divide_in(const LogarithmicTable& other, const Dims& dims) {
  impl().divide_in_dims(other, dims);
  return *this;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::pow(T x) const {
  LogarithmicTable result;
  impl().power(x, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::weighted_update(const LogarithmicTable& other, T x) const {
  LogarithmicTable result;
  impl().weighted_update(other, x, result);
  return result;
}

template <typename T>
Exp<T> LogarithmicTable<T>::maximum(DiscreteValues* values) const {
  return impl().maximum(values);
}

template <typename T>
Exp<T> LogarithmicTable<T>::minimum(DiscreteValues* values) const {
  return impl().minimum(values);
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_front(unsigned n) const {
  LogarithmicTable result;
  impl().maximum_front(n, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_back(unsigned n) const {
  LogarithmicTable result;
  impl().maximum_back(n, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::maximum_dims(const Dims& dims) const {
  LogarithmicTable result;
  impl().maximum_dims(dims, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_front(unsigned n) const {
  LogarithmicTable result;
  impl().minimum_front(n, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_back(unsigned n) const {
  LogarithmicTable result;
  impl().minimum_back(n, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::minimum_dims(const Dims& dims) const {
  LogarithmicTable result;
  impl().minimum_dims(dims, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_front(const DiscreteValues& values) const {
  LogarithmicTable result;
  impl().restrict_front(values, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_back(const DiscreteValues& values) const {
  LogarithmicTable result;
  impl().restrict_back(values, result);
  return result;
}

template <typename T>
LogarithmicTable<T> LogarithmicTable<T>::restrict_dims(const Dims& dims, const DiscreteValues& values) const {
  LogarithmicTable result;
  impl().restrict_dims(dims, values, result);
  return result;
}

template <typename T>
T LogarithmicTable<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T LogarithmicTable<T>::cross_entropy(const LogarithmicTable& other) const {
  return impl().cross_entropy(other);
}

template <typename T>
T LogarithmicTable<T>::kl_divergence(const LogarithmicTable& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
T LogarithmicTable<T>::sum_diff(const LogarithmicTable& other) const {
  return impl().sum_difference(other);
}

template <typename T>
T LogarithmicTable<T>::max_diff(const LogarithmicTable& other) const {
  return impl().max_difference(other);
}

template <typename T>
ProbabilityTable<T> LogarithmicTable<T>::probability() const {
  ProbabilityTable<T> result;
  transform(param(), ExponentOp<T>(), result.param());
  return result;
}

template <typename T>
LogarithmicVector<T> LogarithmicTable<T>::vector() const {
  assert(arity() == 1);
  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(param().data(), size());
}

template <typename T>
LogarithmicMatrix<T> LogarithmicTable<T>::matrix() const {
  assert(arity() == 2);
  using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  return Eigen::Map<const Array>(param().data(), param().size(0), param().size(1));
}

template <typename T>
typename LogarithmicTable<T>::Impl& LogarithmicTable<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *static_cast<Impl*>(impl_.get());
}

template <typename T>
const typename LogarithmicTable<T>::Impl& LogarithmicTable<T>::impl() const {
  assert(impl_);
  return *static_cast<const Impl*>(impl_.get());
}

} // namespace libgm
