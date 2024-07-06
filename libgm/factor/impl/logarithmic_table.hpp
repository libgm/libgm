#pragma once

#include "../logarithmic_table.hpp"

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
struct LogarithmicTable<T>::Impl {
  Table<T> param;

  // Object operations
  //--------------------------------------------------------------------------

  void equals(const Object& other) const override {
    return param == impl(other).param;
  }

  void print(std::ostream& out) const override {
    out << param;
  }

  void save(oarchive& ar) const {
    ar << param;
  }

  void load(iarchive& ar) {
    ar >> param;
  }

  // Assignment
  //--------------------------------------------------------------------------

  void assign(const Exp<T>& x) {
    param.reset({}, x.lv);
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(const Exp<T>& x) const {
    return std::make_unique<Impl>(transform(param, IncrementedBy<T>(x.lv)));
  }

  ImplPtr divide(const Exp<T>& x) const {
    return std::make_unique<Impl>(transform(param, DecrementedBy<T>(x.lv));)
  }

  ImplPtr divide_inverse(const Exp<T>& x) const {
    return std::make_unique<Impl>(transform(param, SubtractedFrom<T>(x.lv)));
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, std::plus<T>()));
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, std::minus<T>()));
  }

  void multiply_in(const Exp<T>& x) {
    param.transform(IncrementedBy<T>(x.lv));
  }

  void divide_in(const Exp<T>& x) {
    param.transform(DecrementedBy<T>(x.lv));
  }

  void multiply_in(const Object& other) {
    param.transform(impl(other).param, std::plus<T>());
  }

  void divide_in(const Object& other) {
    param.transform(impl(other).param, std::minus<T>());
  }

  // Join operations
  //--------------------------------------------------------------------------

  template <typename Op>
  ImplPtr join_front(const Object& other, Op op) const {
    return std::make_unique<Impl>(libgm::join_front(param, impl(other).param, op));
  }

  template <typename Op>
  ImplPtr join_back(const Object& other, Op op) const {
    return std::make_unique<Impl>(libgm::join_back(param, impl(other).param, op));
  }

  template <typename Op>
  ImplPtr join(const Object& other, const Dims& i, const Dims& j, Op op) const {
    return std::make_unique<Impl>(join(param, impl(other).param, i, j, std::plus<T>()));
  }

  ImplPtr multiply_front(const Object& other) const {
    return join_front(other, std::plus<T>());
  }

  ImplPtr multiply_back(const Object& other) const {
    return join_back(other, std::plus<T>());
  }

  ImplPtr multiply(const Object& other, const Dims& i, const Dims& j) const {
    return join(other, i, j, std::plus<T>());
  }

  ImplPtr divide_front(const Object& other) const {
    return join_front(other, std::minus<T>());
  }

  ImplPtr divide_back(const Object& other) const {
    return join_back(other, std::minus<T>());
  }

  ImplPtr divide(const Object& other, const Dims& i, const Dims& j) const {
    return join(other, i, j, std::minus<T>());
  }

  void multiply_in_front(const Object& other) {
    param.join_front(impl(other).param, std::plus<T>());
  }

  void multiply_in_back(const Object& other) {
    param.join_back(impl(other).param, std::plus<T>());
  }

  void multiply_in(const Object& other, const Dims& dims) {
    param.join(impl(other).param, dims, std::plus<T>());
  }

  void divide_in_front(const Object& other) {
    param.join_front(impl(other).param, std::minus<T>());
  }

  void divide_in_back(const Object& other) {
    param.join_back(impl(other).param, std::minus<T>());
  }

  void divide_in(const Object& other, const Dims& dims) {
    param.join(impl(other).param, dims, std::minus<T>());
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return std::make_unique<Impl>(transform(param, MultipliedBy<T>(x)));
  }

  ImplPtr weighted_update(const Object& other, T x) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, WeightedPlus<T>(1 - x, x)));
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    T offset = maximum().lv;
    T sum = param.accumulate(T(0), PlusExponent<T>(-offset));
    return Exp<T>(std::log(sum) + offset);
  }

  Exp<T> maximum(Values* values) const {
    auto it = std::max_element(param.begin(), param.end());
    if (values) {
      param.offset().vector(it - param.begin(), values->resize<size_t>(param.arity()));
    }
    return Exp<T>(*it);
  }

  Exp<T> minimum(Values* values) const {
    auto it = std::min_element(param.begin(), param.end());
    if (values) {
      param.offset().vector(it - param.begin(), values->resize<size_t>(param.arity()));
    }
    return Exp<T>(*it);
  }

  template <typename Op>
  ImplPtr aggregate_front(Op op, T init, unsigned n) const {
    return std::make_unique<Impl>(param.aggregate_front(op, init, n));
  }

  template <typename Op>
  ImplPtr aggregate_back(Op op, T init, unsigned n) const {
    return std::make_unique<Impl>(param.aggregate_back(op, init, n));
  }

  template <typename Op>
  ImplPtr aggregate(Op op, T init, const Dims& retain) const {
    return std::make_unique<Impl>(param.aggregate(op, init, retain));
  }

  ImplPtr marginal_front(unsigned n) const {
    return aggregate_front(LogPlusExp<T>(), -inf<T>(), n);
  }

  ImplPtr marginal_back(unsigned n) const {
    return aggregate_back(LogPlusExp<T>(), -inf<T>(), n);
  }

  ImplPtr marginal(const Dims& retain) const {
    return aggregate(LogPlusExp<T>(), -inf<T>(), retain);
  }

  ImplPtr maximum_front(unsigned n) const {
    return aggregate_front(Maximum<T>(), -inf<T>(), n);
  }

  ImplPtr maximum_back(unsigned n) const {
    return aggregate_back(Maximum<T>(), -inf<T>(), n);
  }

  ImplPtr maximum(const Dims& retain) const {
    return aggregate(Maximum<T>(), -inf<T>(), retain);
  }

  ImplPtr minimum_front(unsigned n) const {
    return aggregate_front(Minimum<T>(), inf<T>(), n);
  }

  ImplPtr minimum_back(unsigned n) const {
    return aggregate_back(Minimum<T>(), inf<T>(), n);
  }

  ImplPtr minimum(const Dims& retain) const {
    return aggregate(Minimum<T>(), inf<T>(), retain);
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    divide_in(marginal());
  }

  ImplPtr conditional(size_t nhead) const {
    // return make_table_function_noalias<log_tag>(
    //   [nhead](const Derived& f, paramtype& result) {
    //     f.param().join_aggregated([](const T* b, const T* e) {
    //         return log_sum_exp(b, e);
    //       }, std::minus<T>(), nhead, result);
    //   }, derived().arity(), derived()
    // );
  }

  // Restrictions
  //--------------------------------------------------------------------------

  ImplPtr restrict_head(const Values& values) const {
    return std::make_unique<Impl>(param.restrict_front(values.size(), values.ptr<size_t>()));
  }

  ImplPtr restrict_tail(const Values& values) const {
    return std::make_unique<Impl>(param.restrict_back(values.size(), values.ptr<size_t>()));
  }

  ImplPtr restrict_dims(const Dims& dims, const Values& values) const {
    assert(dims.count() == values.size());
    return std::make_unique<Impl>(param.restrict(dims, values.ptr<size_t>()));
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return transform_accumulate(param, EntropyLogOp<T>(), std::plus<T>(), T(0));
  }

  T cross_entropy(const Object& other) const {
    return transform_accumulate(param, impl(other).param, EntropyLogOp<T>(), std::plus<T>(), T(0));
  }

  T kl_divergence(const Object& other) const {
    return transform_accumulate(param, impl(other).param, KldLogOp<T>(), std::plus<T>(), T(0));
  }

  T sum_diff(const Object& other) const {
    return transform_accumulate(param, impl(other).param, AbsDifference<T>(), std::plus<T>(), T(0));
  }

  T max_diff(const Object& other) const {
    return transform_accumulate(param, impl(other).param, AbsDifference<T>(), Maximum<T>(), T(0));
  }

#if 0
  // Sampling
  //--------------------------------------------------------------------------

  MultivariateCategoricalDistribution<T> distribution() const {
    return { param, log_tag() };
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
    param.find_if(compose(PartialSumGreaterThan<T>(p), Exponent<T>()), result);
  }
#endif

}; // class LogarithmicTable<T>::Impl

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Exp<T> value) {
  reset({});
  param()[0] = value.lv;
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(const Shape& shape, Exp<T> value) {
  reset(shape);
  param().fill(value.lv);
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(const Shape& shape, std::initializer_list<T> values) {
  reset(shape);
  assert(values.size() == param().size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
LogarithmicTable<T>::LogarithmicTable(Table<T> param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
void LogarithmicTable<T>::reset(const Shape& shape) {
  if (!impl_) {
    impl_.reset(new Impl(shape));
  } else if (param().shape() != shape) {
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
const Shape& LogarithmicTable<T>::shape() const {
  return param().shape();
}

template <typename T>
Table<T>& LogarithmicTable<T>::param() {
  return impl().param;
}

template <typename T>
const Table<T>& LogarithmicTable<T>::param() const {
  return impl().param;
}

template <typename T>
const Exp<T>& LogarithmicTable<T>::operator()(const Assignment& a) const {
  return Exp<T>(param()(a.ptr<size_t>()));
}

template <typename T>
T LogarithmicTable<T>::log(const Assignment& a) const {
  return param()(a.ptr<size_t>())
}

template <typename T>
ProbabilityTable<T> LogarithmicTable<T>::probability() const {
  return ProbabilityTable<T>(transform(param(), Exponent<T>()));
}

template <typename T>
LogarithmicVector<T> LogarithmicTable<T>::vector() const {
  // TODO: use Eigen::Map
}

template <typename T>
LogarithmicMatrix<T> LogarithmicTable<T>::matrix() const {
  // TODO: use Eigen::Map
}

} // namespace libgm
