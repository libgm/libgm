#pragma once

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/math/eigen/dense.hpp>

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
  /// The type representing the sample.
  using result_type = size_t;

  /// Constructor for a distribution in the probability space.
  explicit CategoricalDistribution(const ProbabilityVector<T>& p)
    : psum_(p) {
    std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
  }

  /// Constructor for a distribution in the log space.
  explicit CategoricalDistribution(const LogarithmicVector<T>& p)
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
  Vector<T> psum_;

};

}
