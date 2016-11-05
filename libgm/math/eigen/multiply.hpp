#ifndef LIBGM_EIGEN_MULTIPLY_HPP
#define LIBGM_EIGEN_MULTIPLY_HPP

#include <libgm/traits/int_constant.hpp>

#include <Eigen/Core>

#include <cassert>
#include <utility>

namespace libgm {

  template <typename Matrix, typename Vector, typename Result>
  void mv_multiply(Matrix&& m, Vector&& v, Result&& r, std::size_t /* dim */,
                   int_constant<Eigen::Vertical>) {
    std::forward<Result>(r).noalias() =
      std::forward<Matrix>(m).transpose() * std::forward<Vector>(v);
  }

  template <typename Matrix, typename Vector, typename Result>
  void mv_multiply(Matrix&& m, Vector&& v, Result&& r, std::size_t /* dim */,
                   int_constant<Eigen::Horizontal>) {
    std::forward<Result>(r).noalias() =
      std::forward<Matrix>(m) * std::forward<Vector>(v);
  }

  template <typename Matrix, typename Vector, typename Result>
  void mv_multiply(Matrix&& m, Vector&& v, Result&& r, std::size_t dim,
                   int_constant<Eigen::BothDirections>) {
    assert(dim <= 1);
    if (dim == 0) {
      std::forward<Result>(r).noalias() =
        std::forward<Matrix>(m).transpose() * std::forward<Vector>(v);
    } else {
      std::forward<Result>(r).noalias() =
        std::forward<Matrix>(m) * std::forward<Vector>(v);
    }
  }

} // namespace libgm

#endif
