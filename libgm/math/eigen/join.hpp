#ifndef LIBGM_EIGEN_JOIN_HPP
#define LIBGM_EIGEN_JOIN_HPP

#include <libgm/traits/int_constant.hpp>

#include <Eigen/Core>

#include <cassert>
#include <utility>

namespace libgm {

  template <typename JoinOp, typename Matrix, typename Vector>
  void mv_join(JoinOp join_op, Matrix&& m, Vector&& v, std::size_t /* dim */,
               int_constant<Eigen::Vertical>) {
    join_op(std::forward<Matrix>(m).array().colwise(),
            std::forward<Vector>(v).array());
  }

  template <typename JoinOp, typename Matrix, typename Vector>
  void mv_join(JoinOp join_op, Matrix&& m, Vector&& v, std::size_t /* dim */,
               int_constant<Eigen::Horizontal>) {
    join_op(std::forward<Matrix>(m).array().rowwise(),
            std::forward<Vector>(v).array().transpose());
  }

  template <typename JoinOp, typename Matrix, typename Vector>
  void mv_join(JoinOp join_op, Matrix&& m, Vector&& v, std::size_t dim,
               int_constant<Eigen::BothDirections>) {
    assert(dim <= 1);
    if (dim == 0) {
      join_op(std::forward<Matrix>(m).array().colwise(),
              std::forward<Vector>(v).array());
    } else {
      join_op(std::forward<Matrix>(m).array().rowwise(),
              std::forward<Vector>(v).array().transpose());
    }
  }

  template <typename JoinOp, typename Vector, typename Matrix>
  void vm_join(JoinOp join_op, Vector&& v, Matrix&& m, std::size_t /* dim */,
               int_constant<Eigen::Vertical>) {
    join_op(std::forward<Vector>(v).array().rowwise().replicate(m.cols()),
            std::forward<Matrix>(m).array());
  }

  template <typename JoinOp, typename Vector, typename Matrix>
  void vm_join(JoinOp join_op, Vector&& v, Matrix&& m, std::size_t /* dim */,
               int_constant<Eigen::Horizontal>) {
    join_op(std::forward<Vector>(v).array().rowwise().replicate(m.rows()),
            std::forward<Matrix>(m).array().transpose());
  }

  template <typename JoinOp, typename Vector, typename Matrix>
  void vm_join(JoinOp join_op,  Vector&& v, Matrix&& m, std::size_t dim,
               int_constant<Eigen::BothDirections>) {
    assert(dim <= 1);
    if (dim == 0) {
      join_op(std::forward<Vector>(v).array().rowwise().replicate(m.cols()),
              std::forward<Matrix>(m).array());
    } else {
      join_op(std::forward<Vector>(v).array().rowwise().replicate(m.rows()),
              std::forward<Matrix>(m).array().transpose());
    }
  }

} // namespace libgm

#endif
