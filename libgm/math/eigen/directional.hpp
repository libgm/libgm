#ifndef LIBGM_EIGEN_DIRECTIONAL_HPP
#define LIBGM_EIGEN_DIRECTIONAL_HPP

#include <libgm/math/eigen/dense.hpp>

namespace libgm {

  /**
   * Invokes the given operation on a matrix column-wise.
   */
  template <typename UpdateOp, typename Left, typename Right>
  inline void
  directional_update(UpdateOp update_op,
                     int_constant<Eigen::Vertical>, std::size_t /* dim */,
                     Eigen::ArrayBase<Left>& left,
                     const Eigen::ArrayBase<Right>& right) {
    update_op(left.derived().colwise(), right.derived());
  }

  /**
   * Invokes the given operation on a matrix row-wise.
   */
  template <typename UpdateOp, typename Left, typename Right>
  inline void
  directional_update(UpdateOp update_op,
                     int_constant<Eigen::Horizontal>, std::size_t /* dim */,
                     Eigen::ArrayBase<Left>& left,
                     const Eigen::ArrayBase<Right>& right) {
    update_op(left.derived().rowwise(), right.derived().transpose());
  }

  /**
   * Invokes the given operation on a matrix column-wise or row-wise,
   * as specified by the dimesion.
   */
  template <typename UpdateOp, typename Left, typename Right>
  inline void
  directional_update(UpdateOp update_op,
                     int_constant<Eigen::BothDirections>, std::size_t dim,
                     Eigen::ArrayBase<Left>& left,
                     const Eigen::ArrayBase<Right>& right) {
    if (dim == 0) {
      update_op(left.derived().colwise(), right.derived());
    } else {
      update_op(left.derived().rowwise(), right.derived().transpose());
    }
  }

  /**
   * Eliminates the columns using the given member operation.
   */
  template <typename AggOp, typename Input, typename T>
  inline void
  directional_eliminate(AggOp agg_op,
                        int_constant<Eigen::Vertical>, std::size_t /* dim */,
                        const Eigen::ArrayBase<Input>& input,
                        dense_vector<T>& result) {
    result = agg_op(input.derived().colwise()).transpose();
  }

  /**
   * Computes the log-sum-exp along the columns.
   */
  template <typename Input, typename T>
  inline void
  directional_eliminate(member_logSumExp /* agg_op */,
                        int_constant<Eigen::Vertical>, std::size_t /* dim */,
                        const Eigen::ArrayBase<Input>& input,
                        dense_vector<T>& result) {
    auto&& eval = input.derived().eval();
    T offset = eval.maxCoeff();
    result = (eval - offset).exp().colwise().sum().log().transpose() + offset;
  }

  /**
   * Eliminates the rows using teh given member operation.
   */
  template <typename AggOp, typename Input, typename T>
  inline void
  directional_eliminate(AggOp agg_op,
                        int_constant<Eigen::Horizontal>, std::size_t /* dim */,
                        const Eigen::ArrayBase<Input>& input,
                        dense_vector<T>& result) {
    result = agg_op(input.derived().rowwise());
  }

  /**
   * Computes the log-sum-exp along the rows.
   */
  template <typename Input, typename T>
  inline void
  directional_eliminate(member_logSumExp /* agg_op */,
                        int_constant<Eigen::Horizontal>, std::size_t /* dim */,
                        const Eigen::ArrayBase<Input>& input,
                        dense_vector<T>& result) {
    auto&& eval = input.derived().eval();
    T offset = eval.maxCoeff();
    result = (eval - offset).exp().rowwise().sum().log().transpose() + offset;
  }

  /**
   * Eliminates the dimension specified at runtime using an eliminate operation.
   */
  template <typename AggOp, typename Input, typename T>
  inline void
  directional_eliminate(AggOp agg_op,
                        int_constant<Eigen::BothDirections>, std::size_t dim,
                        const Eigen::ArrayBase<Input>& input,
                        dense_vector<T>& result) {
    if (dim == 0) {
      directional_eliminate(agg_op, int_constant<Eigen::Vertical>(), 0,
                            input, result);
    } else {
      directional_eliminate(agg_op, int_constant<Eigen::Horizontal>(), 0,
                            input, result);
    }
  }

} // namespace libgm

#endif
