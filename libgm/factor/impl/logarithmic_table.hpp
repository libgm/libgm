#ifndef LIBGM_LOGARITHMIC_TABLE_HPP
#define LIBGM_LOGARITHMIC_TABLE_HPP

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
struct LogarithmicTable<T>::Impl {
  Table<T> param_;

  // Object operations
  //--------------------------------------------------------------------------
  void operator==(const LogarithmicTable<T>& other) const override {
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
  void assign(Exp<T> x) {
    param_.reset({}, x.lv);
  }

  // Direct operations
  //--------------------------------------------------------------------------
  LogarithmicTable<T> multiply(Exp<T> x) const {
    return param_.transform(IncrementedBy<T>(x.lv));
  }

  LogarithmicTable<T> divide(Exp<T> x) const {
    return param_.transform(DecrementedBy<T>(x.lv));
  }

  LogarithmicTable<T> divide_inverse(Exp<T> x) const {
    return param_.transform(SubtractedFrom<T>(x.lv));
  }

  LogarithmicTable<T> pow(T x) const {
    return param_.transform(MultipliedBy<T>(x));
  }

  LogarithmicTable<T> add(const LogarithmicTable<T>& other) const {
    return param_.transform(other.impl().param_, LogPlusExp<T>());
  }

  LogarithmicTable<T> multiply(const LogarithmicTable<T>& other) const {
    return param_.transform(other.impl().param_, std::plus<T>());
  }

  LogarithmicTable<T> divide(const LogarithmicTable<T>& other) const {
    return param_.transform(other.impl().param_, std::minus<T>());
  }

  LogarithmicTable<T> max(const LogarithmicTable<T>& other) const {
    return param_.transform(other.impl().param_, Maximum<T>());
  }

  LogarithmicTable<T> min(const LogarithmicTable<T>& other) const {
    return param_.transform(other.impl().param_, Minimum<T>());
  }

  LogarithmicTable<T> weighted_update(const LogarithmicTable<T>& other, T x) const {
    return param_.transform(other.impl().param_, WeightedPlus<T>(1 - x, x));
  }

  // Mutations
  //--------------------------------------------------------------------------

  void multiply_in(Exp<T> x) {
    param_.transform(IncrementedBy<T>(x.lv));
  }

  void divide_in(Exp<T> x) {
    param_.transform(DecrementedBy<T>(x.lv));
  }

  void multiply_in(const LogarithmicTable<T>& other) {
    param_.transform(other.param(), std::plus<T>());
  }

  void divide_in(const LogarithmicTable<T>& other) {
    param_.transform(other.param(), std::minus<T>());
  }

  void multiply_in_front(const LogarithmicTable<T>& other) {
    param_.join_front(other.param(), std::plus<T>());
  }

  void multiply_in_back(const LogarithmicTable<T>& other) {
    param_.join_back(other.param(), std::plus<T>());
  }

  void multiply_in(const LogarithmicTable<T>& other, const DimList& dims) {
    param_.join(other.param(), dims, std::plus<T>());
  }

  void divide_in_front(const LogarithmicTable<T>& other) {
    param_.join_front(other.param(), std::minus<T>());
  }

  void divide_in_back(const LogarithmicTable<T>& other) {
    param_.join_back(other.param(), std::minus<T>());
  }

  void divide_in(const LogarithmicTable<T>& other, const DimList& dims) {
    param_.join(other.param(), dims, std::minus<T>());
  }

  void normalize() {
    *this /= marginal();
  }

  // Joins
  //--------------------------------------------------------------------------
  LogarithmicTable<T> multiply_front(const LogarithmicTable<T>& other) const {
    return join_front(param_, other.param(), std::plus<T>());
  }

  LogarithmicTable<T> multiply_back(const LogarithmicTable<T>& other) const {
    return join_back(param_, other.param(), std::plus<T>());
  }

  LogarithmicTable<T> multiply(const LogarithmicTable<T>& other, const DimList& i, const DimList& j) const {
    return join(param_, other.param(), i, j, std::plus<T>());
  }

  LogarithmicTable<T> divide_front(const LogarithmicTable<T>& other) const {
    return join_front(param_, other.pram(), std::minus<T>());
  }

  LogarithmicTable<T> divide_back(const LogarithmicTable<T>& other) const {
    return join_back(param_, other.pram(), std::minus<T>());
  }

  LogarithmicTable<T> divide(const LogarithmicTable<T>& other, const DimList& i, const DimList& j) const {
    return join(param_, other.param(), i, j, std::minus<T>());
  }

  // Conversions
  //--------------------------------------------------------------------------

  ProbabilityTable<T> probability() const {
    return param_.transform(Exponent<T>());
  }

  LogarithmicVector<T> vector() const {
    // TODO: use Eigen::Map
  }

  LogarithmicMatrix<T> matrix() const {
    // TODO: use Eigen::Map
  }

  // Aggregates
  //--------------------------------------------------------------------------

  LogarithmicTable<T> marginal_front(size_t n) const {
    return param_.aggregate_front(LogPlusExp<T>(), -inf<T>(), n);
  }

  LogarithmicTable<T> marginal_back(size_t n) const {
    return param_.aggregate_back(LogPlusExp<T>(), -inf<T>(), n);
  }

  LogarithmicTable<T> marginal(const DimList& retain) const {
    return param_.aggregate(LogPlusExp<T>(), -inf<T>(), retain);
  }

  LogarithmicTable<T> maximum_front(size_t n) const {
    return param_.aggregate_front(Maximum<T>(), -inf<T>(), n);
  }

  LogarithmicTable<T> maximum_back(size_t n) const {
    return param_.aggregate_back(Maximum<T>(), -inf<T>(), n);
  }

  LogarithmicTable<T> maximum(const DimList& retain) const {
    return param_.aggregate(Maximum<T>(), -inf<T>(), retain);
  }

  LogarithmicTable<T> minimum_front(size_t n) const {
    return param_.aggregate_front(Minimum<T>(), inf<T>(), n);
  }

  LogarithmicTable<T> minimum_back(size_t n) const {
    return param_.aggregate_back(Minimum<T>(), inf<T>(), n);
  }

  LogarithmicTable<T> minimum(const DimList& retain) const {
    return param_.aggregate(Minimum<T>(), inf<T>(), retain);
  }

  Exp<T> marginal() const {
    T offset = maximum().lv;
    T sum = param_.accumulate(T(0), PlusExponent<T>(-offset));
    return Exp<T>(std::log(sum) + offset);
  }

  Exp<T> maximum() const {
    auto it = std::max_element(param_.begin(), param_.end());
    return Exp<T>(*it);
  }

  Exp<T> minimum() const {
    auto it = std::min_element(param_.begin(), param_.end());
    return Exp<T>(*it);
  }

  Exp<T> maximum(Assignment& a) const {
    auto it = std::max_element(param_.begin(), param_.end());
    param_.offset().vector(it - param_.begin(), a);
    return Exp<T>(*it);
  }

  Exp<T> minimum(Assignment& a) const {
    auto it = std::min_element(param_.begin(), param_.end());
    param_.offset().vector(it - param_.begin(), a);
    return Exp<T>(*it);
  }

  bool normalizable() const {
    return maximum().lv > -inf<T>();
  }

  // Conditioning
  //--------------------------------------------------------------------------

  LogarithmicTable<T> conditional(size_t nhead) const {
    // return make_table_function_noalias<log_tag>(
    //   [nhead](const Derived& f, param_type& result) {
    //     f.param().join_aggregated([](const T* b, const T* e) {
    //         return log_sum_exp(b, e);
    //       }, std::minus<T>(), nhead, result);
    //   }, derived().arity(), derived()
    // );
  }

  LogarithmicTable<T> restrict_head(const Assignment& a) const {
    return param_.restrict_front(a.vector<size_t>());
  }

  LogarithmicTable<T> restrict_tail(const Assignment& a) const {
    return param_.restrict_back(a.vector<size_t>());
  }

  LogarithmicTable<T> restrict_list(const DimList& dims, const Assignment& a) const {
    return param_.restrict(dims, a.vector<size_t>());
  }

  // Ordering
  //--------------------------------------------------------------------------

  LogarithmicTable<T> reorder(const DimList& dims) const {
    return param_.reorder(dims);
  }

  // Sampling
  //--------------------------------------------------------------------------

  MultivariateCategoricalDistribution<T> distribution() const {
    return { param_, log_tag() };
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
    auto plus_entropy = compose_right(std::plus<T>(), EntropyLogOp<T>());
    return derived().accumulate(T), plus_entropy);
  }

  T cross_entropy(const LogarithmicTable<T>& other) const {
    return transform_accumulate(EntropyLogOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T kl_divergence(const LogarithmicTable<T>& other) cosnt {
    return transform_accumulate(KldLogOp<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T sum_diff(const LogarithmicTable<T>*& other) const {
    return transform_accumulate(AbsDifference<T>(), std::plus<T>(), T(0), param_, other.impl().param_);
  }

  T max_diff(const LogarithmicTable<T>& other) const {
    return transform_accumulate(AbsDifference<T>(), Maximum<T>(), T(0), param_, other.impl().param_);
  }

}; // class LogarithmicTable<T>::Impl

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Exp<T> value) {
  param().reset();
  param()[0] = value.lv;
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(const std::vector<size_t>& shape, Exp<T> value) {
  reset(shape);
  param().fill(value.lv);
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(const std::vector<size_t>& shape, std::initializer_list<T> values) {
  reset(shape);
  assert(values.size() == param().size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(const Table<T>& param) {
  impl_.reset(new Impl(param));
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Table<T>&& param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
void LogarithmicTable<T>::reset(const std::vector<size_t>& shape) {
  if (param().empty() || param().shape() != shape) {
    param().reset(shape);
  }
}

template <typename T>
size_t LogarithmicTable<T>::arity() const {
  return param().arity();
}

template <typename T>
size_t LogarithmicTable<T>::size() const {
  return param().size();
}

template <typename T>
const std::vector<size_t>& LogarithmicTable<T>::shape() const {
  return param().shape();
}

template <typename T>
T* LogarithmicTable<T>::begin() {
  return param().begin();
}

template <typename T>
const T* LogarithmicTable<T>::begin() const {
  return param().begin();
}

template <typename T>
T* LogarithmicTable<T>::end() {
  return param().end();
}

template <typename T>
const T* LogarithmicTable<T>::end() const {
  return param().end();
}

template <typename T>
Table<T>& LogarithmicTable<T>::param() {
  return impl().param_;
}

template <typename T>
const Table<T>& LogarithmicTable<T>::param() const {
  return impl().param_;
}

template <typename T>
Exp<T> LogarithmicTable<T>::operator()(const Assignment& a) const {
  return Exp<T>(param()(a.ptr<size_t>()));
}

T log(const Assignment& a) const {
  return param()(a.ptr<size_t>())
}

} // namespace libgm

#endif
