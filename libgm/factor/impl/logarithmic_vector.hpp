#pragma once

#include "../logarithmic_vector.hpp"

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/serialization/eigen.hpp>
// #include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
// #include <libgm/math/random/categorical_distribution.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicVector<T>::Impl {

  /// The parameters of the factor, i.e., a vector of log-probabilities.
  DenseVector<T> param;

  // Constructors
  //--------------------------------------------------------------------------

  explicit Impl(size_t size)
    : param(size) {}

  explicit Impl(DenseVector<T> param)
    : param(std::move(param)) {}

  // Utility functions
  //--------------------------------------------------------------------------

  const T* begin() const {
    return param.data();
  }

  const T* end() const {
    return param.data() + param.size();
  }

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const Object& other) const override {
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

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(const Exp<T>& x) const {
    return std::make_unique<Impl>(param + x.lv);
  }

  ImplPtr divide(const Exp<T>& x) const {
    return std::make_unique<Impl>(param - x.lv);
  }

  ImplPtr divide_inverse(const Exp<T>& x) const {
    return std::make_unique<Impl>(x.lv - param);
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(param + impl(other).param);
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(param - impl(other).param);
  }

  void multiply_in(const Exp<T>& x) {
    param.array() += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    param.array() -= x.lv;
  }

  void multiply_in(const Object& other) {
    param.array() += impl(other).param.array();
  }

  void divide_in(const Object& other) {
    param.array() -= impl(other).param.array();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return std::make_unique<Impl>(param * x);
  }

  ImplPtr weighted_udpate(const Object& other, T x) const {
    return std::make_unique<Impl>((1 - x) * param + x * impl(other).param);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    return Exp<T>(param.logSumExp());
  }

  Exp<T> maximum(Values* values) const {
    return Exp<T>(a ? param.maxCoeff(values->set<size_t>(0)) : param.maxCoeff());
  }

  Exp<T> minimum(Assignment* a) const {
    return Exp<T>(a ? param.minCoeff(values->set<size_t>(0)) : param.minCoeff());
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    param -= marginal().lv;
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return *std::transform(begin(), end(), AccumulatingIterator<T>(), EntropyLogOp<T>());
  }

  template <typename Op>
  T transform_sum(const Object& other, Op op) const {
    const Impl& x = *this;
    const Impl& y = impl(other);
    return *std::transform(x.begin(), x.end(), y.begin(), AccumulatingIterator<T>(), op);
  }

  T cross_entropy(const Object& other) const {
    return transform_sum(other, EntropyLogOp<T>());
  }

  T kl_divergence(const Object& other) const {
    return transform_sum(other, KldLogOp<T>());
  }

  T sum_diff(const Object& other) const {
    return abs(param - impl(other).param).sum();
  }

  T max_diff(const Object& other) const {
    return abs(param - impl(other).param).maxCoeff();
  }

  // CategoricalDistribution<T> distribution() const {
  //   return { param, log_tag() };
  // }

}; // struct Impl

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length) {
  reset(length);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length, Exp<T> x) {
  reset(length);
  impl().param.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(DenseVector<T> param)
  : Object(new Impl(std::move(param))) { }

template <typename T>
LogarithmicVector<T>::LogarithmicVector(std::initializer_list<T> params) {
  reset(param.size();)
  std::copy(params.begin(), params.end(), impl().param.data());
}

template <typename T>
void LogarithmicVector<T>::reset(size_t length) {
  if (impl_) {
    impl().param.resize(length);
  } else {
    impl_.reset(new Impl(length));
  }
}

size_t LogarithmicVector<T>::size() const {
  return impl().param.size();
}

template <typename T>
DenseVector<T>& LogarithmicVector<T>::param() {
  return impl().param;
}

template <typename T>
const DenseVector<T>& LogarithmicVector<T>::param() const {
  return impl().param;
}

template <typename T>
T LogarithmicVector<T>::log(size_t row) const {
  return impl().param[row];
}

template <typename T>
T LogarithmicVector<T>::log(const Assignment& a) const {
  return impl().param[a.get<size_t>(0)];
}

template <typename T>
ProbabilityVector<T> LogarithmicVector<T>::probability() const {
  return ProbabilityVector<T>(exp(param()));
}

template <typename T>
LogarithmicTable<T> LogarithmicVector<T>::table() const {
  ...;
}

} // namespace libgm
