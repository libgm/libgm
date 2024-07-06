#pragma once

#include <libgm/datastructure/table.hpp>
#include <libgm/math/likelihood/mle_eval.hpp>

#include <functional>

namespace libgm {

/**
 * A maximum likelihood estimator of a probability table.
 *
 * \tparam RealType the real type representing the parameters
 */
template <typename RealType = double>
class ProbabilityTableMLE {
public:
  /// The regularization parameter.
  typedef RealType regul_type;

  /// The parameters of the distribution computed by this estimator.
  typedef Table<RealType> param_type;

  /**
   * Constructs a maximum-likelihood estimator with the specified
   * regularization parameters.
   */
  ProbabilityTableMLE(const uint_vector& shape, RealType regul = ReaType())
    : shape_(shape), regul_(regul) { }

  /**
   * Computes the maximum likelihood estimate of a probability table
   * from unweighted data.
   */
  Table<RealType>
  operator()(const dense_matrix_ref<size_t>& samples) const {
    Table<RealType> counts = dense_matrix<RealType>::Ones(regul_);
    for (size_t i = 0; i < samples.cols(); ++i) {
      ++counts(samples.col(i));
    }
    counts.normalize();
    return counts;
  }

  /**
   * Computes the maximum likelihood estimate of a probability table
   * from weighted data.
   */
  Table<RealType>
  operator()(const dense_matrix_ref<size_t>& samples,
              const dense_vector_ref<RealType>& weights) const {
    Table<RealType> counts = dense_matrix<RealType>::Ones(regul_);
    for (size_t i = 0; i < samples.cols(); ++i) {
      counts(samples.col(i)) += weights[i];
    }
    counts.normalize();
    return counts;
  }

private:
  /// The shape of the estimated table.
  uint_vector shape_;

  /// The regularization parameter.
  RealType regul_;

}; // class ProbabilityTableMLE

} // namespace libgm
