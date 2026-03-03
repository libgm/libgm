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
// #include <libgm/math/likelihood/canonical_table_ll.hpp>
// #include <libgm/math/random/multivariate_categorical_distribution.hpp>

namespace libgm {

template <typename T>
struct ProbabilityTable<T>::Impl : Object::Impl {
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

  void assign(const T& x) {
    param.reset({});
    param[0] = x;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply(const T& x, ProbabilityTable& result) const {
    transform(param, MultipliedBy<T>(x), result.param());
  }

  void divide(const T& x, ProbabilityTable& result) const {
    transform(param, DividedBy<T>(x), result.param());
  }

  void divide_inverse(const T& x, ProbabilityTable& result) const {
    transform(param, Dividing<T>(x), result.param());
  }

  void multiply(const ProbabilityTable& other, ProbabilityTable& result) const {
    transform(param, other.param(), std::multiplies<T>(), result.param());
  }

  void divide(const ProbabilityTable& other, ProbabilityTable& result) const {
    transform(param, other.param(), std::divides<T>(), result.param());
  }

  void multiply_in(const T& x) {
    transform_in(param, MultipliedBy<T>(x));
  }

  void divide_in(const T& x) {
    transform_in(param, DividedBy<T>(x));
  }

  void multiply_in(const ProbabilityTable& other) {
    transform_in(param, other.param(), std::multiplies<T>());
  }

  void divide_in(const ProbabilityTable& other) {
    transform_in(param, other.param(), std::divides<T>());
  }

  // Join operations
  //--------------------------------------------------------------------------

  void multiply_front(const ProbabilityTable& other, ProbabilityTable& result) const {
    join_front(param, other.param(), std::multiplies<T>(), result.param());
  }

  void multiply_back(const ProbabilityTable& other, ProbabilityTable& result) const {
    join_back(param, other.param(), std::multiplies<T>(), result.param());
  }

  void multiply_dims(const ProbabilityTable& other, const Dims& i, const Dims& j, ProbabilityTable& result) const {
    join(param, other.param(), i, j, std::multiplies<T>(), result.param());
  }

  void divide_front(const ProbabilityTable& other, ProbabilityTable& result) const {
    join_front(param, other.param(), std::divides<T>(), result.param());
  }

  void divide_back(const ProbabilityTable& other, ProbabilityTable& result) const {
    join_back(param, other.param(), std::divides<T>(), result.param());
  }

  void divide_dims(const ProbabilityTable& other, const Dims& i, const Dims& j, ProbabilityTable& result) const {
    join(param, other.param(), i, j, std::divides<T>(), result.param());
  }

  void multiply_in_front(const ProbabilityTable& other) {
    join_in_front(param, other.param(), std::multiplies<T>());
  }

  void multiply_in_back(const ProbabilityTable& other) {
    join_in_back(param, other.param(), std::multiplies<T>());
  }

  void multiply_in_dims(const ProbabilityTable& other, const Dims& dims) {
    join_in(param, other.param(), dims, std::multiplies<T>());
  }

  void divide_in_front(const ProbabilityTable& other) {
    join_in_front(param, other.param(), std::divides<T>());
  }

  void divide_in_back(const ProbabilityTable& other) {
    join_in_back(param, other.param(), std::divides<T>());
  }

  void divide_in_dims(const ProbabilityTable& other, const Dims& dims) {
    join_in(param, other.param(), dims, std::divides<T>());
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  void power(T x, ProbabilityTable& result) const {
    transform(param, PowerOp<T>(x), result.param());
  }

  void weighted_update(const ProbabilityTable& other, T x, ProbabilityTable& result) const {
    transform(param, other.param(), WeightedPlus<T>(1 - x, x), result.param());
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return std::accumulate(begin(), end(), T(0), std::plus<T>());
  }

  T maximum(DiscreteValues* values) const {
    auto it = std::max_element(begin(), end());
    if (values) {
      *values = param.shape().index(it - begin());
    }
    return *it;
  }

  T minimum(DiscreteValues* values) const {
    auto it = std::min_element(begin(), end());
    if (values) {
      *values = param.shape().index(it - begin());
    }
    return *it;
  }

  void marginal_front(unsigned n, ProbabilityTable& result) const {
    aggregate_front(param, n, T(0), std::plus<T>(), result.param());
  }

  void marginal_back(unsigned n, ProbabilityTable& result) const {
    aggregate_back(param, n, T(0), std::plus<T>(), result.param());
  }

  void marginal_dims(const Dims& retain, ProbabilityTable& result) const {
    aggregate(param, retain, T(0), std::plus<T>(), result.param());
  }

  void maximum_front(unsigned n, ProbabilityTable& result) const {
    aggregate_front(param, n, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void maximum_back(unsigned n, ProbabilityTable& result) const {
    aggregate_back(param, n, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void maximum_dims(const Dims& retain, ProbabilityTable& result) const {
    aggregate(param, retain, -inf<T>(), MaximumOp<T>(), result.param());
  }

  void minimum_front(unsigned n, ProbabilityTable& result) const {
    aggregate_front(param, n, inf<T>(), MinimumOp<T>(), result.param());
  }

  void minimum_back(unsigned n, ProbabilityTable& result) const {
    aggregate_back(param, n, inf<T>(), MinimumOp<T>(), result.param());
  }

  void minimum_dims(const Dims& retain, ProbabilityTable& result) const {
    aggregate(param, retain, inf<T>(), MinimumOp<T>(), result.param());
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    divide_in(marginal());
  }

  void normalize(unsigned nhead) {
    // return make_table_function_noalias<log_tag>(
    //   [nhead](const Derived& f, paramtype& result) {
    //     f.param().join_aggregated([](const T* b, const T* e) {
    //         return log_sum_exp(b, e);
    //       }, std::minus<T>(), nhead, result.param());
    //   }, derived().arity(), derived()
    // );
  }

  // Restrictions
  //--------------------------------------------------------------------------

  void restrict_front(const DiscreteValues& values, ProbabilityTable& result) const {
    libgm::restrict_front(param, values.vec(), result.param());
  }

  void restrict_back(const DiscreteValues& values, ProbabilityTable& result) const {
    libgm::restrict_back(param, values.vec(), result.param());
  }

  void restrict_dims(const Dims& dims, const DiscreteValues& values, ProbabilityTable& result) const {
    libgm::restrict(param, dims, values.vec(), result.param());
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return std::accumulate(begin(), end(), T(0), [](T accu, T val) {
      return accu + EntropyOp<T>()(val);
    });
  }

  template <typename AccuOp, typename TransOp>
  T transform_accumulate(const ProbabilityTable& other, AccuOp accu, TransOp trans) const {
    assert(param.shape() == other.shape());
    return std::inner_product(begin(), end(), other.impl().begin(), T(0), accu, trans);
  }

  T cross_entropy(const ProbabilityTable& other) const {
    return transform_accumulate(other, std::plus<T>(), EntropyOp<T>());
  }

  T kl_divergence(const ProbabilityTable& other) const {
    return transform_accumulate(other, std::plus<T>(), KldOp<T>());
  }

  T sum_difference(const ProbabilityTable<T>& other) const {
    return transform_accumulate(other, std::plus<T>(), AbsDifference<T>());
  }

  T max_difference(const ProbabilityTable& other) const {
    return transform_accumulate(other, MaximumOp<T>(), AbsDifference<T>());
  }

}; // class ProbabilityTable<T>::Impl

template <typename T>
ProbabilityTable<T>::ProbabilityTable(T value)
  : Object(std::make_unique<Impl>()) {
  param()[0] = value;
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, T value)
  : Object(std::make_unique<Impl>(std::move(shape))){
  param().fill(value);
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, std::initializer_list<T> values)
  : Object(std::make_unique<Impl>(std::move(shape))) {
  assert(values.size() == size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Shape shape, const T* values)
  : Object(std::make_unique<Impl>(std::move(shape))) {
  std::copy(values, values + size(), param().begin());
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Table<T> param)
  : Object(std::make_unique<Impl>(std::move(param))) {}

template <typename T>
size_t ProbabilityTable<T>::arity() const {
  return param().arity();
}

template <typename T>
size_t ProbabilityTable<T>::size() const {
  return param().size();
}

template <typename T>
const Shape& ProbabilityTable<T>::shape() const {
  return param().shape();
}

template <typename T>
Table<T>& ProbabilityTable<T>::param() {
  if (!impl_) {
    impl_.reset(new Impl);
  }
  return impl().param;
}

template <typename T>
const Table<T>& ProbabilityTable<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbabilityTable<T>::operator()(const DiscreteValues& values) const {
  return param()(values.vec());
}

template <typename T>
T ProbabilityTable<T>::log(const DiscreteValues& values) const {
  return std::log(param()(values.vec()));
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator*(T x) const {
  ProbabilityTable result;
  impl().multiply(x, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator*(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().multiply(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator*=(T x) {
  impl().multiply_in(x);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator*=(const ProbabilityTable& other) {
  impl().multiply_in(other);
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator/(T x) const {
  ProbabilityTable result;
  impl().divide(x, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_inverse(T x) const {
  ProbabilityTable result;
  impl().divide_inverse(x, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::operator/(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().divide(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator/=(T x) {
  impl().divide_in(x);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::operator/=(const ProbabilityTable& other) {
  impl().divide_in(other);
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply_front(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().multiply_front(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply_back(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().multiply_back(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::multiply(const ProbabilityTable& other, const Dims& i, const Dims& j) const {
  ProbabilityTable result;
  impl().multiply_dims(other, i, j, result);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in_front(const ProbabilityTable& other) {
  impl().multiply_in_front(other);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in_back(const ProbabilityTable& other) {
  impl().multiply_in_back(other);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::multiply_in(const ProbabilityTable& other, const Dims& dims) {
  impl().multiply_in_dims(other, dims);
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_front(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().divide_front(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide_back(const ProbabilityTable& other) const {
  ProbabilityTable result;
  impl().divide_back(other, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::divide(const ProbabilityTable& other, const Dims& i, const Dims& j) const {
  ProbabilityTable result;
  impl().divide_dims(other, i, j, result);
  return result;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in_front(const ProbabilityTable& other) {
  impl().divide_in_front(other);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in_back(const ProbabilityTable& other) {
  impl().divide_in_back(other);
  return *this;
}

template <typename T>
ProbabilityTable<T>& ProbabilityTable<T>::divide_in(const ProbabilityTable& other, const Dims& dims) {
  impl().divide_in_dims(other, dims);
  return *this;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::pow(T x) const {
  ProbabilityTable result;
  impl().power(x, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::weighted_update(const ProbabilityTable& other, T x) const {
  ProbabilityTable result;
  impl().weighted_update(other, x, result);
  return result;
}

template <typename T>
T ProbabilityTable<T>::marginal() const {
  return impl().marginal();
}

template <typename T>
T ProbabilityTable<T>::maximum(DiscreteValues* values) const {
  return impl().maximum(values);
}

template <typename T>
T ProbabilityTable<T>::minimum(DiscreteValues* values) const {
  return impl().minimum(values);
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_front(unsigned n) const {
  ProbabilityTable result;
  impl().marginal_front(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_back(unsigned n) const {
  ProbabilityTable result;
  impl().marginal_back(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::marginal_dims(const Dims& retain) const {
  ProbabilityTable result;
  impl().marginal_dims(retain, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_front(unsigned n) const {
  ProbabilityTable result;
  impl().maximum_front(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_back(unsigned n) const {
  ProbabilityTable result;
  impl().maximum_back(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::maximum_dims(const Dims& retain) const {
  ProbabilityTable result;
  impl().maximum_dims(retain, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_front(unsigned n) const {
  ProbabilityTable result;
  impl().minimum_front(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_back(unsigned n) const {
  ProbabilityTable result;
  impl().minimum_back(n, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::minimum_dims(const Dims& retain) const {
  ProbabilityTable result;
  impl().minimum_dims(retain, result);
  return result;
}

template <typename T>
void ProbabilityTable<T>::normalize() {
  impl().normalize();
}

template <typename T>
void ProbabilityTable<T>::normalize_head(unsigned nhead) {
  impl().normalize(nhead);
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_front(const DiscreteValues& values) const {
  ProbabilityTable result;
  impl().restrict_front(values, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_back(const DiscreteValues& values) const {
  ProbabilityTable result;
  impl().restrict_back(values, result);
  return result;
}

template <typename T>
ProbabilityTable<T> ProbabilityTable<T>::restrict_dims(const Dims& dims, const DiscreteValues& values) const {
  ProbabilityTable result;
  impl().restrict_dims(dims, values, result);
  return result;
}

template <typename T>
T ProbabilityTable<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T ProbabilityTable<T>::cross_entropy(const ProbabilityTable& other) const {
  return impl().cross_entropy(other);
}

template <typename T>
T ProbabilityTable<T>::kl_divergence(const ProbabilityTable& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
T ProbabilityTable<T>::sum_diff(const ProbabilityTable& other) const {
  return impl().sum_difference(other);
}

template <typename T>
T ProbabilityTable<T>::max_diff(const ProbabilityTable& other) const {
  return impl().max_difference(other);
}

template <typename T>
LogarithmicTable<T> ProbabilityTable<T>::logarithmic() const {
  LogarithmicTable<T> result;
  transform(param(), LogarithmOp<T>(), result.param());
  return result;
}

template <typename T>
ProbabilityVector<T> ProbabilityTable<T>::vector() const {
  assert(arity() == 1);
  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(param().data(), size());
}

template <typename T>
ProbabilityMatrix<T> ProbabilityTable<T>::matrix() const {
  assert(arity() == 2);
  using Array = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
  return Eigen::Map<const Array>(param().data(), param().size(0), param().size(1));
}

template <typename T>
typename ProbabilityTable<T>::Impl& ProbabilityTable<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *static_cast<Impl*>(impl_.get());
}

template <typename T>
const typename ProbabilityTable<T>::Impl& ProbabilityTable<T>::impl() const {
  assert(impl_);
  return *static_cast<const Impl*>(impl_.get());
}

} // namespace libgm
