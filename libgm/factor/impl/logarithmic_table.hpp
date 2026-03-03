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
const typename LogarithmicTable<T>::VTable LogarithmicTable<T>::vtable{
  &LogarithmicTable<T>::Impl::multiply,
  &LogarithmicTable<T>::Impl::multiply,
  &LogarithmicTable<T>::Impl::multiply_in,
  &LogarithmicTable<T>::Impl::multiply_in,
  &LogarithmicTable<T>::Impl::divide,
  &LogarithmicTable<T>::Impl::divide_inverse,
  &LogarithmicTable<T>::Impl::divide,
  &LogarithmicTable<T>::Impl::divide_in,
  &LogarithmicTable<T>::Impl::divide_in,
  &LogarithmicTable<T>::Impl::multiply_front,
  &LogarithmicTable<T>::Impl::multiply_back,
  &LogarithmicTable<T>::Impl::multiply_dims,
  &LogarithmicTable<T>::Impl::multiply_in_front,
  &LogarithmicTable<T>::Impl::multiply_in_back,
  &LogarithmicTable<T>::Impl::multiply_in_dims,
  &LogarithmicTable<T>::Impl::divide_front,
  &LogarithmicTable<T>::Impl::divide_back,
  &LogarithmicTable<T>::Impl::divide_dims,
  &LogarithmicTable<T>::Impl::divide_in_front,
  &LogarithmicTable<T>::Impl::divide_in_back,
  &LogarithmicTable<T>::Impl::divide_in_dims,
  &LogarithmicTable<T>::Impl::power,
  &LogarithmicTable<T>::Impl::weighted_update,
  &LogarithmicTable<T>::Impl::maximum,
  &LogarithmicTable<T>::Impl::minimum,
  &LogarithmicTable<T>::Impl::maximum_front,
  &LogarithmicTable<T>::Impl::maximum_back,
  &LogarithmicTable<T>::Impl::maximum_dims,
  &LogarithmicTable<T>::Impl::minimum_front,
  &LogarithmicTable<T>::Impl::minimum_back,
  &LogarithmicTable<T>::Impl::minimum_dims,
  &LogarithmicTable<T>::Impl::restrict_front,
  &LogarithmicTable<T>::Impl::restrict_back,
  &LogarithmicTable<T>::Impl::restrict_dims,
  &LogarithmicTable<T>::Impl::entropy,
  &LogarithmicTable<T>::Impl::cross_entropy,
  &LogarithmicTable<T>::Impl::kl_divergence,
  &LogarithmicTable<T>::Impl::sum_difference,
  &LogarithmicTable<T>::Impl::max_difference,
};

} // namespace libgm
