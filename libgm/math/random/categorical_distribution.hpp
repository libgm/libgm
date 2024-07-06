#pragma once

#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/tags.hpp>

#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

/**
 * A categorical distribution over a single variable,
 * whose parameters are represented by a dense Eigen vector.
 */
template <typename T = double>
class CategoricalDistribution {
public:
  /// The underlying parameter type.
  typedef DenseVector<T> param_type;

  /// The type representing the sample.
  typedef size_t result_type;

  /// Constructor for a distribution in the probability space.
  CategoricalDistribution(const DenseVector<T>& p, prob_tag)
    : psum_(p) {
    std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
  }

  /// Constructor for a distribution in the log space.
  CategoricalDistribution(const DenseVector<T>& p, log_tag)
    : psum_(exp(p.array())) {
    std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
  }

  /// Draws a random sample from a marginal distribution.
  template <typename Generator>
  size_t operator()(Generator& rng) const {
    const T* begin = psum_.data();
    T p = std::uniform_real_distribution<T>()(rng);
    std::ptrdiff_t i
      = std::upper_bound(begin, begin + psum_.size(), p) - begin;
    if (i < psum_.size()) {
      return i;
    } else {
      throw std::invalid_argument("The probabilities are less than 1");
    }
  }

private:
  /// Partial sums.
  DenseVector<T> psum_;

}; // class CategoricalDistribution

} // namespace libgm
