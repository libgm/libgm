#pragma once

#include "../probability_vector.hpp"

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/serialization/eigen.hpp>

#include <numeric>

namespace libgm {

template <typename T>
struct ProbabilityVector<T>::Impl {

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

  ImplPtr multiply(const T& x) const {
    return std::make_unique<Impl>(param * x);
  }

  ImplPtr divide(const T& x) const {
    return std::make_unique<Impl>(param / x);
  }

  ImplPtr divide_inverse(const T& x) const {
    return std::make_unique<Impl>(x / param);
  }

  ImplPtr multiply(const Object& other) const {
    return std::make_unique<Impl>(param * impl(other).param);
  }

  ImplPtr divide(const Object& other) const {
    return std::make_unique<Impl>(param / impl(other).param);
  }

  void multiply_in(const T& x) {
    param.array() *= x;
  }

  void divide_in(const T& x) {
    param.array() /= x;
  }

  void multiply_in(const Object& other) {
    param.array() *= impl(other).param.array();
  }

  void divide_in(const Object& other) {
    param.array() /= impl(other).param.array();
  }

  // Arithmetic
  //--------------------------------------------------------------------------

  ImplPtr pow(T x) const {
    return std::make_unique<Impl>(param.pow(x));
  }


  ImplPtr weighted_udpate(const Object& other, T x) const {
    return std::make_unique<Impl>((1 - x) * param + x * impl(other).param);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  T marginal() const {
    return param.sum();
  }

  T maximum(Values* values) const {
    return a ? param.maxCoeff(values->set<size_t>(0)) : param.maxCoeff();
  }

  T minimum(Values* values) const {
    return a ? param.minCoeff(values->set<size_t>(0)) : param.minCoeff();
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    param() /= marginal();
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    return *std::transform(begin(), end(), AccumulatingIterator<T>(), EntropyOp<T>());
  }

  template <typename Op>
  T transform_sum(const Object& other, Op op) const {
    const Impl& x = *this;
    const Impl& y = impl(other);
    return *std::transform(x.begin(), x.end(), y.begin(), AccumulatingIterator<T>(), op);
  }

  T cross_entropy(const Object& other) const {
    return transform_sum(other, EntropyOp<T>());
  }

  T kl_divergence(const Object& other) const {
    return transform_sum(other, KldOp<T>());
  }

  T sum_diff(const Object& other) const {
    return abs(param - impl(other).param).sum();
  }

  T max_diff(const Object& other) const {
    return abs(param - impl(other).param).maxCoeff();
  }

  // CategoricalDistribution<T> distribution() const {
  //   return param;
  // }

}; // struct Impl

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length) {
  reset(length);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(size_t length, T x) {
  reset(length);
  impl().param.fill(x);
}

template <typename T>
ProbabilityVector<T>::ProbabilityVector(DenseVector<T> param)
  : Implements(new Impl(std::move(param))) { }

template <typename T>
ProbabilityVector<T>::ProbabilityVector(std::initializer_list<T> params) {
  reset(param.size();)
  std::copy(params.begin(), params.end(), impl().param.data());
}

template <typename T>
void ProbabilityVector<T>::reset(size_t length) {
  if (impl_) {
    impl().param.resize(length);
  } else {
    impl_.reset(new Impl(length));
  }
}

template <typename T>
size_t ProbabilityVector<T>::size() const {
  return impl().param.size();
}

template <typename T>
DenseVector<T>& ProbabilityVector<T>::param() {
  return impl().param;
}

template <typename T>
const DenseVector<T>& ProbabilityVector<T>::param() const {
  return impl().param;
}

template <typename T>
T ProbabilityVector<T>::operator()(size_t row) const {
  return impl().param[row];
}

template <typename T>
T ProbabilityVector<T>::operator()(const Assignment& a) const {
  return impl().param[a.get<size_t>(0)];
}

template <typename T>
T ProbabilityVector<T>::log(size_t row) const {
  return std::log(impl().param[row]);
}

template <typename T>
T ProbabilityVector<T>::log(const Assignment& a) const {
  return std::log(impl().param[a.get<size_t>(0)]);
}

template <typename T>
LogarithmicVector<T> ProbabilityVector<T>::logarithmic() const {
  return param().log();
}

template <typename T>
ProbabilityTable<T> ProbabilityVector<T>::table() const {
  return ...;
}

} // namespace libgm
