#ifndef LIBGM_MATRIX_VIEW_HPP
#define LIBGM_MATRIX_VIEW_HPP

#include <libgm/factor/experimental/expression/matrix_base.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/tuple.hpp>

namespace libgm { namespace experimental {

  /**
   * A matrix expression that returns a view of other matrix / vector
   * expressions.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix (prob_tag or log_tag).
   * \tparam Op
   *         A functor that given all F, returns an Eigen matrix expression.
   * \tparam F
   *         A (possibly empty) parameter pack of matrix or vector expressions.
   */
  template <typename Space, typename Op, typename... F>
  class matrix_view
    : public matrix_base<
        Space,
        typename std::result_of_t<Op(F...)>::Scalar,
        matrix_view<Space, Op, F...> > {
  public:
    // Shortcuts
    using real_type = typename std::result_of_t<Op(F...)>::Scalar;

    matrix_view(Op op, const F&... f)
      : op_(op), data_(f...) { }

    bool alias(const real_vector<real_type>& param) const {
      return tuple_any(make_member_alias(param), data_);
    }

    bool alias(const real_matrix<real_type>& param) const {
      return tuple_any(make_member_alias(param), data_);
    }

    void eval_to(real_matrix<real_type>& result) const {
      result = param();
    }

    auto param() const {
      return tuple_apply(op_, data_);
    }

  private:
    Op op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class matrix_view

  /**
   * Creates a matrix_view object, automatically deducing its type.
   * \relates matrix_view
   */
  template <typename Space, typename Op, typename... F>
  inline matrix_view<Space, Op, F...>
  make_matrix_view(Op op, const F&... f) {
    return { op, f... };
  }

  /**
   * Returns a matrix_view expression object representing a view of a raw
   * buffer with known number of rows and columns.
   * \relates matrix_view
   */
  template <typename Space, typename RealType>
  inline auto
  matrix_raw(const RealType* ptr, std::size_t rows, std::size_t cols) {
    return make_matrix_view<Space>(
      [ptr, rows, cols]() {
        return Eigen::Map<const real_matrix<RealType> >(ptr, rows, cols);
      });
  }

} } // namespace libgm::experimental

#endif
