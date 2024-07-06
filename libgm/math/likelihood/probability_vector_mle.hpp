#pragma once

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

namespace libgm {

/**
 * A maximum likelihood estimator for a vector in the probability space.
 *
 * \tparam RealType the real type representing the parameters
 */
template <typename RealType = double>
class ProbabilityVectorMLE {
public:
  /// The regularization parameter type.
  typedef RealType regul_type;

  /// The parameters of the distribution computed by this estimator.
  typedef DenseVector<RealType> param_type;

  /**
   * Constructs a maximum likelihood estimator with the specified
   * regularization parameters.
   */
  ProbabilityVectorMLE(size_t n, RealType regul = RealType(0))
    : n_(n), regul_(regul) { }

  /**
   * Computes the maximum likelihood estimate of a probability vector
   * from unweighted data.
   */
  DenseVector<RealType>
  operator()(const dense_vector_ref<size_t>& samples) const {
    DenseVector<RealType> counts;
    counts.setConstant(n_, regul_);
    for (ptrdiff_t i = 0; i < samples.size(); ++i) {
      ++counts[samples[i]];
    }
    counts /= counts.sum();
    return counts;
  }

  /**
   * Comptues the maximum likelihood estimate of a probability vector
   * from weighted data.
   */
  DenseVector<RealType>
  operator()(const dense_vector_ref<size_t>& samples,
              const dense_vector_ref<RealType>& weights) const {
    assert(samples.size() == weights.size());
    DenseVector<RealType> counts;
    counts.setConstant(n_, regul_);
    for (ptrdiff_t i = 0; i < samples.size(); ++i) {
      counts[samples[i]] += wieghts[i];
    }
    counts /= counts.sum();
    return counts;
  }

private:
  /// The number of rows of the estimated vector.
  size_t n_;

  /// The regularization parameter.
  RealType regul_;

}; // class ProbabilityVectorMLE

} // namespace libgm
