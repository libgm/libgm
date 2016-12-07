#ifndef LIBGM_MATRIX_TRANSFORM_HPP
#define LIBGM_MATRIX_TRANSFORM_HPP

#include <libgm/factor/expression/matrix_base.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>

namespace libgm { namespace experimental {

  /**
   * A class that represents an element-wise transform of one or more matrices.
   * The matrices must have the same number of rows and columns.
   *
   * \tparam Space
   *         A tag denoting the space of the matrix, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(F) dense matrices
   *         and returns a dense matrix expression.
   * \tparam F
   *         A non-empty pack of matrix expressions.
   */
  template <typename Space, typename Op, typename... F>
  class matrix_transform
    : public matrix_base<
        Space,
        typename std::result_of_t<
          Op(decltype(std::declval<F>().param().array())...)>::Scalar,
        matrix_transform<Space, Op, F...> > {

  public:
    // shortcuts
    using real_type  = typename std::result_of_t<
      Op(decltype(std::declval<F>().param().array())...)>::Scalar;

    using param_type = dense_matrix<real_type>;

    matrix_transform(Op op, const std::tuple<const F&...> data)
      : op_(op), data_(data) { }

    bool alias(const dense_vector<real_type>& param) const {
      return false;
      // matrix_transform never aliases a vector because the param()
    }

    bool alias(const dense_matrix<real_type>& param) const {
      // this is not completely correct -- we do not detect e.g. when
      // one of the components is a transpose
      return false;
    }

    void eval_to(param_type& result) const {
      result = expr(tuple_transform(member_param(), data_));
    }

    template <typename AssignOp>
    void transform_inplace(AssignOp assign_op, param_type& result) const {
      assign_op(result.array(), expr(tuple_transform(member_param(), data_)));
    }

    template <typename AccuOp>
    real_type accumulate(AccuOp op) const {
      return op(expr(tuple_transform(member_param(), data_)));
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_transform<ResultSpace, compose_t<UnaryOp, Op>, F...>
    transform(UnaryOp unary_op) const {
      return { compose(unary_op, op_), data_ };
    }

    friend Op transform_op(const matrix_transform& f) {
      return f.op_;
    }

    friend std::tuple<const F&...> transform_data(const matrix_transform& f) {
      return f.data_;
    }

  private:
    /**
     * Returns the Eigen expression representing the result of this transform.
     * Because some of the matrices may be temporaries, member_param() must be
     * invoked in the caller and not here.
     */
    template <typename... Matrices>
    auto expr(const std::tuple<Matrices...>& matrices) const {
      return tuple_apply(op_, tuple_transform(member_array(), matrices));
    }

    Op op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;

  }; // class matrix_transform

  /**
   * Constructs a matrix_transform object, deducing its type.
   * \relates matrix_transform
   */
  template <typename Space, typename Op, typename... F>
  inline matrix_transform<Space, Op, F...>
  make_matrix_transform(Op op, const std::tuple<const F&...> data) {
    return { op, data };
  }

  /**
   * The default transform associated with a matrix expression.
   * \relates matrix_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline identity
  transform_op(const matrix_base<Space, RealType, Derived>& f) {
    return identity();
  }

  /**
   * The default transform data assciated with a matrix expression.
   * \relates matrix_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline std::tuple<const Derived&>
  transform_data(const matrix_base<Space, RealType, Derived>& f) {
    return std::tie(f.derived());
  }

  /**
   * Transforms two matrices with identical Space and RealType.
   * \relates matrix_base
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op,
            const matrix_base<Space, RealType, F>& f,
            const matrix_base<Space, RealType, G>& g) {
    auto fdata = transform_data(f.derived());
    auto gdata = transform_data(g.derived());
    constexpr std::size_t m = std::tuple_size<decltype(fdata)>::value;
    constexpr std::size_t n = std::tuple_size<decltype(gdata)>::value;
    return make_matrix_transform<Space>(
      compose<m, n>(binary_op, transform_op(f.derived()), transform_op(g.derived())),
      std::tuple_cat(fdata, gdata)
    );
  }

} } // namespace libgm::experimental

#endif
