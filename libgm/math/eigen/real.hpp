#ifndef LIBGM_EIGEN_REAL_HPP
#define LIBGM_EIGEN_REAL_HPP

#include <libgm/traits/int_constant.hpp>
#include <libgm/traits/vector_value.hpp>

#include <Eigen/Core>

#include <numeric>

namespace libgm {

  template <typename T = double>
  using real_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <typename T = double>
  using real_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  //! Specialization of vector_value for real_vector.
  template <typename T>
  struct vector_value<real_vector<T>> {
    typedef T type;
  };

  /**
   * Performs a binary transform of two vectors and accumulates the result.
   */
  template <typename TransOp, typename AggOp, typename T>
  T transform_accumulate(TransOp trans_op, AggOp agg_op, T init,
                         const real_vector<T>& p, const real_vector<T>& q) {
    assert(p.rows() == q.rows());
    return std::inner_product(p.data(), p.data() + p.size(), q.data(), T(0),
                              agg_op, trans_op);
  }

  /**
   * Performs a binary transform of two matrices and accumulates the result.
   */
  template <typename TransOp, typename AggOp, typename T>
  T transform_accumulate(TransOp trans_op, AggOp agg_op, T init,
                         const real_matrix<T>& p, const real_matrix<T>& q) {
    assert(p.rows() == q.rows() && p.cols() == q.cols());
    return std::inner_product(p.data(), p.data() + p.size(), q.data(), T(0),
                              agg_op, trans_op);
  }

} // namespace libgm

#endif
