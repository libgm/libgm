#pragma once

#include "../probability_table.hpp"

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
struct ProbabilityTable<T>::Impl {
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

  void assign(const T& x) {
    param.reset({}, x);
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(const T& x) const {
    return std::make_unique<Impl>(transform(param, MultipliedBy<T>(x)));
  }

  ImplPtr divide(const T& x) const {
    return std::make_unique<Impl>(transform(param, DividedBy<T>(x)));
  }

  ImplPtr divide_inverse(const T& x) const {
    return std::make_unique<Impl>(transform(param, Dividing<T>(x)));
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, std::multiplies<T>()));
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, std::divides<T>()));
  }

  void multiply_in(const T& x) {
    param.transform(MultipliedBy<T>(x));
  }

  void divide_in(const T& x) {
    param.transform(DividedBy<T>(x));
  }

  void multiply_in(const Object& other) {
    param.transform(impl(other).param(), std::multiplies<T>());
  }

  void divide_in(const Object& other) {
    param.transform(impl(other).param(), std::divides<T>());
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
    return join_front(other, std::multiplies<T>());
  }

  ImplPtr multiply_back(const Object& other) const {
    return join_back(other, std::multiplies<T>());
  }

  ImplPtr multiply(const Object& other, const Dims& i, const Dims& j) const {
    return join(other, i, j, std::multiplies<T>());
  }

  ImplPtr divide_front(const Object& other) const {
    return join_front(other, std::divides<T>());
  }

  ImplPtr divide_back(const Object& other) const {
    return join_back(other, std::divides<T>());
  }

  ImplPtr divide(const Object& other, const Dims& i, const Dims& j) const {
    return join(other, i, j, std::divides<T>());
  }

  void multiply_in_front(const Object& other) {
    param.join_front(impl(other).param(), std::multiplies<T>());
  }

  void multiply_in_back(const Object& other) {
    param.join_back(impl(other).param(), std::multiplies<T>());
  }

  void multiply_in(const Object& other, const Dims& dims) {
    param.join(impl(other).param(), dims, std::multiplies<T>());
  }

  void divide_in_front(const Object& other) {
    param.join_front(impl(other).param(), std::divides<T>());
  }

  void divide_in_back(const Object& other) {
    param.join_back(impl(other).param(), std::divides<T>());
  }

  void divide_in(const Object& other, const Dims& dims) {
    param.join(impl(other).param(), dims, std::divides<T>());
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return std::make_unique<Impl>(transform(param, Pow<T>(x)));
  }

  ImplPtr weighted_update(const Object& other, T x) const {
    return std::make_unique<Impl>(transform(param, impl(other).param, WeightedPlus<T>(1 - x, x)));
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param.accumulate(T(0), std::plus<T>());
  }

  T maximum(Values* values) const {
    auto it = std::max_element(param.begin(), param.end());
    if (values) {
      param.offset().vector(it - param.begin(), values->resize<size_t>(param.arity()));
    }
    return *it;
  }

  T minimum(Assignment* a) const {
    auto it = std::min_element(param.begin(), param.end());
    if (values) {
      param.offset().vector(it - param.begin(), values->resize<size_t>(param.arity()));
    }
    return *it;
  }

  ImplPtr marginal_front(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_front(std::sum<T>(), T(0), n));
  }

  ImplPtr marginal_back(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_back(std::sum<T>(), T(0), n));
  }

  ImplPtr marginal(const Dims& retain) const {
    return std::make_unique<Impl>(param.aggregate(std::sum<T>(), T(0), retain));
  }

  ImplPtr maximum_front(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_front(Maximum<T>(), -inf<T>(), n));
  }

  ImplPtr maximum_back(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_back(Maximum<T>(), -inf<T>(), n));
  }

  ImplPtr maximum(const Dims& retain) const {
    return std::make_unique<Impl>(param.aggregate(Maximum<T>(), -inf<T>(), retain));
  }

  ImplPtr minimum_front(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_front(Minimum<T>(), inf<T>(), n));
  }

  ImplPtr minimum_back(size_t n) const {
    return std::make_unique<Impl>(param.aggregate_back(Minimum<T>(), inf<T>(), n));
  }

  ImplPtr minimum(const Dims& retain) const {
    return std::make_unique<Impl>(param.aggregate(Minimum<T>(), inf<T>(), retain));
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    divide_in(marginal());
  }

  void normalize(size_t nhead) const {
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

  ImplPtr restrict_list(const Dims& dims, const Values& values) const {
    assert(dims.count() == values.size());
    return std::make_unique<Impl>(param.restrict(dims, values.ptr<size_t>()));
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return transform_accumulate(param, EntropyOp<T>(), std::plus<T>(), T(0));
  }

  T cross_entropy(const Object& other) const {
    return transform_accumulate(param, impl(other).param, EntropyOp<T>(), std::plus<T>(), T(0));
  }

  T kl_divergence(const Object& other) cosnt {
    return transform_accumulate(param, impl(other).param, KldOp<T>(), std::plus<T>(), T(0));
  }

  T sum_diff(const ProbabilityTable<T>*& other) const {
    return transform_accumulate(param, impl(other).param, AbsDifference<T>(), std::plus<T>(), T(0));
  }

  T max_diff(const Object& other) const {
    return transform_accumulate(param, impl(other).param, AbsDifference<T>(), Maximum<T>(), T(0));
  }

#if 0
  // Sampling
  //--------------------------------------------------------------------------

  MultivariateCategoricalDistribution<T> distribution() const {
    return param;
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
}; // class ProbabilityTable<T>::Impl

template <typename T>
ProbabilityTable<T>::ProbabilityTable(T value) {
  reset({});
  param()[0] = value;
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(const std::vector<size_t>& shape, T value) {
  reset(shape);
  param().fill(value);
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(const Shape& shape, std::initializer_list<T> values) {
  reset(shape);
  assert(values.size() == param().size());
  std::copy(values.begin(), values.end(), param().begin());
}

template <typename T>
ProbabilityTable<T>::ProbabilityTable(Table<T> param) {
  impl_.reset(new Impl(std::move(param)));
}

template <typename T>
void ProbabilityTable<T>::reset(const Shape& shape) {
  if (!impl_) {
    impl_.reset(new Impl(shape));
  } else if (param().shape() != shape) {
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
const Shape& ProbabilityTable<T>::shape() const {
  return param().shape();
}

template <typename T>
Table<T>& ProbabilityTable<T>::param() {
  return impl().param;
}

template <typename T>
const Table<T>& ProbabilityTable<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbabilityTable<T>::operator()(const Assignment& a) const {
  return param()(a.ptr<size_t>());
}

T log(const Assignment& a) const {
  return std::log(param()(a.ptr<size_t>()));
}

LogarithmicTable<T> ProbabilityTable<T>::logarithmic() const {
  return LogarithmicTable<T>(param.transform(Logarithm<T>()));
}

ProbabilityVector<T> ProbabilityTable<T>::vector() const {
  // TODO: use Eigen::Map
}

ProbabilityMatrix<T> ProbabilityTable<T>::matrix() const {
  // TODO: use Eigen::Map
}

} // namespace libgm

#endif
