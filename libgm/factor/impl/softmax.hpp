#pragma once

#include "../softmax.hpp"

#include <libgm/math/random/softmax_distribution.hpp>

namespace libgm {

template <typename T>
struct Softmax<T>::Impl : Object::Impl {
  /// The shape of the tail arguments.
  Shape tail_shape;

  /// The weight matrix.
  DenseMatrix<T> weight;

  /// The bias vector.
  DenseVector<T> bias;

  Impl(Shape tail_shape, DenseMatrix<T> weight, DenseMatrix<T> bias)
    : tail_shape(std::move(tail_shape)), weight(std::move(weight)), bias(std::move(bias)) {
    check_param();
  }

  /**
   * Checks if the dimensions of the parameters match this factor's arguments.
   * \throw std::invalid_argument if some of the dimensions do not match.
   */
  void check_param() const {
    if (weight.rows() != bias.rows()) {
      throw std::invalid_argument("Inconsistent number of labels.");
    }
    if (weight.cols() != tail_shape.sum()) {
      throw std::invalid_argument("Inconsistent number of features.");
    }
  }

  // Object functions
  //==========================================================================
  void save(oarchive& ar) const override {
    ar << tail_shape << weight << bias;
  }

  void load(iarchive& ar) override {
    ar >> tail_shape >> weight >> bias;
    assert(weight.rows() == bias.rows());
  }

  void print(std::ostream& out) const override {
    out << "Softmax(" << tail_shape << std::endl
        << weight << std::endl
        << bias << ")";
  }

  // Restrictions
  //==========================================================================

  ImplPtr restrict_tail(const Values& values) {
    assert(values.size() == weight.cols());
    DenseVector<T> y = weigth * values.vector<T>() + bias;
    y = (y.array() - y.maxCoeff()).exp();
    y /= y.sum();
    return std::make_unique<ProbabilityVector<T>::Impl>(std::move(y));
    // TODO: sparse discrete vector
  }

#if 0
  // Sampling
  //==========================================================================

  /// Returns the distribution with the parameters of this factor.
  softmax_distribution<T> distribution() const {
    return softmax_distribution<T>(param_);
  }

  /// Draws a random sample from a conditional distribution.
  template <typename Generator>
  size_t sample(Generator& rng, const DenseVector<T>& tail) const {
    return param_.sample(rng, tail);
  }
#endif
};

template <typename T>
Softmax<T>::Softmax(Shape tail_shape, DenseMatrix<T> weight, DenseVector<T> bias)
  : Implements(std::make_unique<Impl>(std::move(tail_shape), std::move(weight), std::move(bias))) {}

template <typename T>
size_t Softmax<T>::tail_arity() const {
  return impl().tail_shape.size();
}

template <typename T>
size_t Softmax<T>::arity() const {
  return impl().tail_shape.size() + 1;
}

template <typename T>
size_t Softmax<T>::num_labels() const {
  return impl().weight.cols();
}

template <typename T>
const Shape& Softmax<T>::tail_shape() const {
  return impl().tail_shape;
}

template <typename T>
const DenseMatrix<T>& Softmax<T>::weight() const {
  return impl().weight;
}

template <typename T>
const DenseVector<T>& Softmax<T>::bias() const {
  return impl().bias;
}

template <typename T>
const DenseMatrix<T>& Softmax<T>::weight() const {
  return impl().weight;
}

template <typename T>
T Softmax<T>::operator()(size_t label, const Values& features) const {
  DenseVector<T> y = impl().weight * features.vector<T>() + impl().bias;
  y = (y.array() - y.maxCoeff()).exp();
  return y[label] / y.sum();
}

template <typename T>
T Softmax<T>::log(size_t label, const Values& features) const {
  return std::log(operator()(label, features));
}

} // namespace libgm
