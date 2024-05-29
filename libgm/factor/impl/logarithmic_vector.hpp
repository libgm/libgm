#ifndef LIBGM_LOGARITHMIC_VECTOR_HPP
#define LIBGM_LOGARITHMIC_VECTOR_HPP

#include <libgm/argument/domain.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/entropy.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/logarithmic.hpp>
#include <libgm/serialization/eigen.hpp>
#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

template <typename T>
struct LogarithmicVector<T>::Impl {

  /// The parameters of the factor, i.e., a vector of log-probabilities.
  DenseVector<T> param_;

  // Object functions
  //--------------------------------------------------------------------------

  bool equals(const LogarithmicVector& other) const override {
    return param_ == other.impl().param_;
  }

  void print(std::ostream& out) const override {
    out << param_;
  }

  void save(oarchive& ar) const {
    ar << param_;
  }

  void load(iarchive& ar) {
    ar >> param_;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply_in(const Exp<T>& x) {
    param_.array() += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    param_.array() -= x.lv;
  }

  void multiply_in(const LogarithmicVector<T>& other) {
    param_.array() += other.impl().param_.array();
  }

  void divide_in(const LogarithmicVector<T>& other) {
    param_.array() -= other.impl().param_.array();
  }

  void normalize() {
    *this /= this->sum();
  }

  LogarithmicVector<T> multiply(Exp<T> x) const {
    return param_ + x.lv;
  }

  LogarithmicVector<T> divide(Exp<T> x) const {
    return param_ - x.lv;
  }

  LogarithmicVector<T> divide_inverse(Exp<T> x) const {
    return x.lv - param_;
  }

  LogarithmicVector<T> pow(T x) const {
    return param_ * x;
  }

  LogarithmicVector<T> add(const LogarithmicVector<T>& other) {
    return log_plus_exp(param_. other.impl().param_);
  }

  LogarithmicVector<T> multiply(const LogarithmicVector<T>& other) const {
    return param_ + other.impl().param_;
  }

  LogarithmicVector<T> divide(const LogarithmicVector<T>& other) const {
    return param_ - other.impl().param_;
  }

  LogarithmicVector<T> weighted_udpate(const LogarithmicVector<T>& other, T x) const {
    return (1 - x) * param_ + x * other.impl().param_;
  }

  // Conversions
  //-----------------------------------------------

  ProbabilityVector<T> probability() const {
    return exp(param_);
  }

  LogarithmicTable<T> table() const {
    ...;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    return Exp<T>(param_.logSumExp());
  }

  Exp<T> max() const {
    return Exp<T>(param_.maxCoeff());
  }

  Exp<T> min() const {
    return Exp<T>(param_.minCoeff());
  }

  Exp<T> max(std::size_t& row) const {
    return Exp<T>(param_.maxCoeffIndex(&row));
  }

  Exp<T> min(std::size_t& row) const {
    return Exp<T>(param_.minCoeffIndex(&row));
  }

  bool normalizable() const {
    return max().lv > -inf<T>();
  }

  CategoricalDistribution<T> distribution() const {
    return { param_, log_tag() };
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    auto plus_entropy =
      compose_right(std::plus<T>(), entropy_log_op<T>());
    return std::accumulate(param_.data(), param_.data() + param_.size(), T(0), plus_entropy);
  }

  T cross_entropy(const LogarithmicVector<T>& other) const {
    return transform_accumulate(
      entropy_log_op<T>(), std::plus<T>(), T(0),
      p.impl().param_, q.impl().param_
    );
  }

  T kl_divergence(const LogarithmicVector<T>& other) const {
    return transform_accumulate(
      kld_log_op<T>(), std::plus<T>(), T(0),
      p.impl().param_, q.impl().param_
    );
  }

  T sum_diff(const LogarithmicVector<T>& other) const {
    return abs(param_ - other.impl().param_).sum();
  }

  T max_diff(const LogarithmicVector<T>& other) const {
    return abs(param_ - other.impl().param_).maxCoeff();
  }
}; // struct Impl

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length) {
  reset(length);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(size_t length, Exp<T> x) {
  reset(length);
  impl().param_.fill(x.lv);
}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(const DenseVector<T>& param)
  : Object(new Impl(param)) {}

template <typename T>
LogarithmicVector<T>::LogarithmicVector(DenseVector<T>&& param)
  : Object(new Impl(std::move(param))) { }

template <typename T>
LogarithmicVector<T>::LogarithmicVector(std::initializer_list<T> params) {
  reset(param.size();)
  std::copy(params.begin(), params.end(), impl().param_.data());
}

template <typename T>
void LogarithmicVector<T>::reset(size_t length) {
  if (impl_) {
    impl().param.resize(length);
  } else {
    impl_.reset(new Impl(length));
  }
}

template <typename T>
size_t LogarithmicVector<T>::size() const {
  return impl().param_.size();
}

template <typename T>
T* LogarithmicVector<T>::begin() {
  return impl().param_.data();
}

template <typename T>
const T* LogarithmicVector<T>::begin() const {
  return impl().param_.data();
}

template <typename T>
T* LogarithmicVector<T>::end() {
  return impl().param_.data() + impl().param_.size();
}

template <typename T>
const T* LogarithmicVector<T>::end() const {
  return impl().param_.data() + impl().param_.size();
}

template <typename T>
DenseVector<T>& LogarithmicVector<T>::param() {
  return impl().param_;
}

template <typename T>
const DenseVector<T>& LogarithmicVector<T>::param() const {
  return impl().param_;
}

template <typename T>
T LogarithmicVector<T>::log(size_t row) const {
  return impl().param_[row];
}

template <typename T>
T LogarithmicVector<T>::log(const Assignment& a) const {
  return impl().param_[a.get<size_t>(0)];
}

} // namespace libgm

#endif
