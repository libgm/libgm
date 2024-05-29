#pragma once

#include <libgm/datastructure/table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/math/numeric.hpp>
#include <libgm/math/likelihood/canonical_table_ll.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/math/tags.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm {

template <typename T>
struct ProbabilityTable<T>::Impl {
  Table<T> param_;

  // Object operations
  //--------------------------------------------------------------------------
  void operator==(const ProbabilityTable<T>& other) const override {
    return param_ == other.impl().param_;
  }

  void print(std::ostream& out) const override {
    out << f.derived().param();
  }

  void save(oarchive& ar) const {
    ar << param_;
  }

  void load(iarchive& ar) {
    ar >> param_;
  }

  // Assignment
  //--------------------------------------------------------------------------
  void assign(T x) {
    param_.reset({}, x);
  }

  // Direct operations
  //--------------------------------------------------------------------------
  ProbabilityTable<T> multiply(T x) const {
    return param_.transform(MultipliedBy<T>(x));
  }

  ProbabilityTable<T> divide(T x) const {
    return param_.transform(DividedBy<T>(x));
  }

  ProbabilityTable<T> divide_inverse(T x) const {
    return param_.transform(Dividing<T>(x));
  }

  ProbabilityTable<T> pow(T x) const {
    return param_.transform(Pow<T>(x));
  }

  ProbabilityTable<T> add(const ProbabilityTable<T>& other) const {
    return param_.transform(other.impl().param_, std::sum<T>());
  }

  ProbabilityTable<T> multiply(const ProbabilityTable<T>& other) const {
    return param_.transform(other.impl().param_, std::multiplies<T>());
  }

  ProbabilityTable<T> divide(const ProbabilityTable<T>& other) const {
    return param_.transform(other.impl().param_, std::divides<T>());
  }

  ProbabilityTable<T> max(const ProbabilityTable<T>& other) const {
    return param_.transform(other.impl().param_, Maximum<T>());
  }

  ProbabilityTable<T> min(const ProbabilityTable<T>& other) const {
    return param_.transform(other.impl().param_, Minimum<T>());
  }

  ProbabilityTable<T> weighted_update(const ProbabilityTable<T>& other, T x) const {
    return param_.transform(other.impl().param_, WeightedPlus<T>(1 - x, x));
  }

  // Mutations
  //--------------------------------------------------------------------------

  void multiply_in(T x) {
    param_.transform(MultipliedBy<T>(x));
  }

  void divide_in(T x) {
    param_.transform(DividedBy<T>(x));
  }

  void multiply_in(const ProbabilityTable<T>& other) {
    param_.transform(other.param(), std::multiplies<T>());
  }

  void divide_in(const ProbabilityTable<T>& other) {
    param_.transform(other.param(), std::divides<T>());
  }

  void multiply_in_front(const ProbabilityTable<T>& other) {
    param_.join_front(other.param(), std::multiplies<T>());
  }

  void multiply_in_back(const ProbabilityTable<T>& other) {
    param_.join_back(other.param(), std::multiplies<T>());
  }

  void multiply_in(const ProbabilityTable<T>& other, const DimList& dims) {
    param_.join(other.param(), dims, std::multiplies<T>());
  }

  void divide_in_front(const ProbabilityTable<T>& other) {
    param_.join_front(other.param(), std::divides<T>());
  }

  void divide_in_back(const ProbabilityTable<T>& other) {
    param_.join_back(other.param(), std::divides<T>());
  }

  void divide_in(const ProbabilityTable<T>& other, const DimList& dims) {
    param_.join(other.param(), dims, std::divides<T>());
  }

  void normalize() {
    *this /= marginal();
  }

  // Joins
  //--------------------------------------------------------------------------
  ProbabilityTable<T> multiply_front(const ProbabilityTable<T>& other) const {
    return join_front(param_, other.param(), std::multiplies<T>());
  }

  ProbabilityTable<T> multiply_back(const ProbabilityTable<T>& other) const {
    return join_back(param_, other.param(), std::multiplies<T>());
  }

  ProbabilityTable<T> multiply(const ProbabilityTable<T>& other, const DimList& i, const DimList& j) const {
    return join(param_, other.param(), i, j, std::multiplies<T>());
  }

  ProbabilityTable<T> divide_front(const ProbabilityTable<T>& other) const {
    return join_front(param_, other.pram(), std::divides<T>());
  }

  ProbabilityTable<T> divide_back(const ProbabilityTable<T>& other) const {
    return join_back(param_, other.pram(), std::divides<T>());
  }

  ProbabilityTable<T> divide(const ProbabilityTable<T>& other, const DimList& i, const DimList& j) const {
    return join(param_, other.param(), i, j, std::divides<T>());
  }

  // Conversions
  //--------------------------------------------------------------------------

  LogarithmicTable<T> logarithmic() const {
    return param_.transform(Logarithm<T>());
  }

  ProbabilityVector<T> vector() const {
    // TODO: use Eigen::Map
  }

  ProbabilityMatrix<T> matrix() const {
    // TODO: use Eigen::Map
  }

  // Aggregates
  //--------------------------------------------------------------------------

  ProbabilityTable<T> marginal_front(size_t n) const {
    return param_.aggregate_front(std::sum<T>(), T(), n);
  }

  ProbabilityTable<T> marginal_back(size_t n) const {
    return param_.aggregate_back(std::sum<T>(), T(), n);
  }

  ProbabilityTable<T> marginal(const DimList& retain) const {
    return param_.aggregate(std::sum<T>(), T(), retain);
  }

  ProbabilityTable<T> maximum_front(size_t n) const {
    return param_.aggregate_front(Maximum<T>(), -inf<T>(), n);
  }

  ProbabilityTable<T> maximum_back(size_t n) const {
    return param_.aggregate_back(Maximum<T>(), -inf<T>(), n);
  }

  ProbabilityTable<T> maximum(const DimList& retain) const {
    return param_.aggregate(Maximum<T>(), -inf<T>(), retain);
  }

  ProbabilityTable<T> minimum_front(size_t n) const {
    return param_.aggregate_front(Minimum<T>(), inf<T>(), n);
  }

  ProbabilityTable<T> minimum_back(size_t n) const {
    return param_.aggregate_back(Minimum<T>(), inf<T>(), n);
  }

  ProbabilityTable<T> minimum(const DimList& retain) const {
    return param_.aggregate(Minimum<T>(), inf<T>(), retain);
  }

  T marginal() const {
    return param_.accumulate(T(0), std::plus<T>());
  }

  T maximum() const {
    auto it = std::max_element(param_.begin(), param_.end());
    return T(*it);
  }

  T minimum() const {
    auto it = std::min_element(param_.begin(), param_.end());
    return T(*it);
  }

  T maximum(Assignment& a) const {
    auto it = std::max_element(param_.begin(), param_.end());
    param_.offset().vector(it - param_.begin(), a);
    return T(*it);
  }

  T minimum(Assignment& a) const {
    auto it = std::min_element(param_.begin(), param_.end());
    param_.offset().vector(it - param_.begin(), a);
    return T(*it);
  }

  bool normalizable() const {
    return maximum() > T(0);
  }

  // Conditioning
  //--------------------------------------------------------------------------

  ProbabilityTable<T> conditional(size_t nhead) const {
    // return make_table_function_noalias<log_tag>(
    //   [nhead](const Derived& f, param_type& result) {
    //     f.param().join_aggregated([](const T* b, const T* e) {
    //         return log_sum_exp(b, e);
    //       }, std::minus<T>(), nhead, result);
    //   }, derived().arity(), derived()
    // );
  }

  ProbabilityTable<T> restrict_head(const Assignment& a) const {
    return param_.restrict_front(a.vector<size_t>());
  }

  ProbabilityTable<T> restrict_tail(const Assignment& a) const {
    return param_.restrict_back(a.vector<size_t>());
  }

  ProbabilityTable<T> restrict_list(const DimList& dims, const Assignment& a) const {
    return param_.restrict(dims, a.vector<size_t>());
  }

  // Ordering
  //--------------------------------------------------------------------------

  ProbabilityTable<T> reorder(const DimList& dims) const {
    return param_.reorder(dims);
  }

  // Sampling
  //--------------------------------------------------------------------------

  MultivariateCategoricalDistribution<T> distribution() const {
    return param_;
  }

  /**
    * Draws a random sample from a marginal distribution represented by this
    * expression.
    *
    * \throw std::out_of_range
    *        may be thrown if the distribution is not normalized
    */
  template <typename Generator>
  uint_vector sample(Generator& rng) const {
    uint_vector result; sample(rng, result); return result;
  }

  /**
    * Draws a random sample from a marginal distribution represented by this
    * expression, storing the result in an output vector.
    *
    * \throw std::out_of_range
    *        may be thrown if the distribution is not normalized
    */
  template <typename Generator>
  void sample(Generator& rng, uint_vector& result) const {
    T p = std::uniform_real_distribution<T>()(rng);
    param_.find_if(compose(PartialSumGreaterThan<T>(p), Exponent<T>()), result);
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    auto plus_entropy = compose_right(std::plus<T>(), EntropyOp<T>());
    return derived().accumulate(T), plus_entropy);
  }

  T cross_entropy(const ProbabilityTable<T>& other) const {
    return transform_accumulate(EntropyOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T kl_divergence(const ProbabilityTable<T>& other) cosnt {
    return transform_accumulate(KldOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T sum_diff(const ProbabilityTable<T>*& other) const {
    return transform_accumulate(AbsDifference<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T max_diff(const ProbabilityTable<T>& other) const {
    return transform_accumulate(AbsDifference<T>(), Maximum<T>(), T(0), param_, other.impl().param_);
  }

}; // class ProbabilityTable<T>::Impl

template <typename T>
ProbabilityTable<T>::ProbabilityTable(T value) {
  param().reset();
  param()[0] = value;
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(const std::vector<size_t>& shape, T value) {
  reset(shape);
  param().fill(value);
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(const std::vector<size_t>& shape, std::initializer_list<T> values) {
  reset(shape);
  assert(values.size() == param().size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(const Table<T>& param) {
  impl_.reset(new Impl(param));
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Table<T>&& param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
void ProbabilityTable<T>::reset(const std::vector<size_t>& shape) {
  if (param().empty() || param().shape() != shape) {
    param().reset(shape);
  }
}

template <typename T>
size_t ProbabilityTable<T>::arity() const {
  return param().arity();
}

template <typename T>
size_t ProbabilityTable<T>::size() const {
  return param().size();
}

template <typename T>
const std::vector<size_t>& ProbabilityTable<T>::shape() const {
  return param().shape();
}

template <typename T>
T* ProbabilityTable<T>::begin() {
  return param().begin();
}

template <typename T>
const T* ProbabilityTable<T>::begin() const {
  return param().begin();
}

template <typename T>
T* ProbabilityTable<T>::end() {
  return param().end();
}

template <typename T>
const T* ProbabilityTable<T>::end() const {
  return param().end();
}

template <typename T>
Table<T>& ProbabilityTable<T>::param() {
  return impl().param_;
}

template <typename T>
const Table<T>& ProbabilityTable<T>::param() const {
  return impl().param_;
}

template <typename T>
T ProbabilityTable<T>::operator()(const Assignment& a) const {
  return param()(a.ptr<size_t>());
}

T log(const Assignment& a) const {
  return std::log(param()(a.ptr<size_t>()));
}

} // namespace libgm

#endif
