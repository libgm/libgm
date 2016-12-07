#ifndef LIBGM_VECTOR_TRANSFORM_HPP
#define LIBGM_VECTOR_TRANSFORM_HPP

#include <libgm/factor/expression/vector_base.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/functional/tuple.hpp>

namespace libgm { namespace experimental {

  /**
   * A class represents an element-wise transform of one or more vectors.
   * The vectors must have the same number of rows.
   *
   * \tparam Space
   *         A tag denoting the space of the vector, e.g., prob_tag or log_tag.
   * \tparam Op
   *         A function object that accepts sizeof...(F) dense vectors
   *         and returns a dense vector expression.
   * \tparam F
   *         A non-empty pack of vector expressions.
   */
  template <typename Space, typename Op, typename... F>
  class vector_transform
    : public vector_base<
        Space,
        typename std::result_of_t<
          Op(decltype(std::declval<F>().param().array())...)>::Scalar,
        vector_transform<Space, Op, F...> > {

  public:
    // shortcuts
    using real_type  = typename std::result_of_t<
      Op(decltype(std::declval<F>().param().array())...)>::Scalar;
    using param_type = dense_vector<real_type>;

    vector_transform(Op op, const F&... f)
      : op_(op), data_(f...) { }

    vector_transform(Op op, const std::tuple<const F&...>& data)
      : op_(op), data_(data) { }

    bool alias(const dense_vector<real_type>& param) const {
      return false;
      // vector_transform is always safe to evaluate, because it is performed
      // element-wise on the input arrays, and input arrays are trivial
    }

    bool alias(const dense_matrix<real_type>& param) const {
      return false;
      // vector_transform evaluates to a vector temporary, so it cannot alias
      // a matrix
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
    vector_transform<ResultSpace, compose_t<UnaryOp, Op>, F...>
    transform(UnaryOp unary_op) const {
      return { compose(unary_op, op_), data_ };
    }

    friend Op transform_op(const vector_transform& f) {
      return f.op_;
    }

    friend std::tuple<const F&...> transform_data(const vector_transform& f) {
      return f.data_;
    }

  private:
    /**
     * Returns the Eigen expression representing the result of this transform.
     * Because some of the vectors may be temporaries, member_param() must be
     * invoked in the caller and not here.
     */
    template <typename... Vectors>
    auto expr(const std::tuple<Vectors...>& vectors) const {
      return tuple_apply(op_, tuple_transform(member_array(), vectors));
    }

    Op op_;
    std::tuple<add_const_reference_if_factor_t<F>...> data_;
  }; // class vector_transform

  /**
   * Constructs a vector_transform object, deducing its type.
   * \relates vector_transform
   */
  template <typename Space, typename Op, typename... F>
  inline vector_transform<Space, Op, F...>
  make_vector_transform(Op op, const std::tuple<const F&...>& data) {
    return { op, data };
  }

  /**
   * The default transform associated with a vector expression.
   * \relates vector_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline identity
  transform_op(const vector_base<Space, RealType, Derived>& f) {
    return identity();
  }

  /**
   * The default transform data assciated with a vector expression.
   * \relates vector_base
   */
  template <typename Space, typename RealType, typename Derived>
  inline std::tuple<const Derived&>
  transform_data(const vector_base<Space, RealType, Derived>& f) {
    return std::tie(f.derived());
  }

  /**
   * Transforms two vector with identical Space and RealType.
   * \relates vector_base
   */
  template <typename BinaryOp, typename Space, typename RealType,
            typename F, typename G>
  inline auto
  transform(BinaryOp binary_op,
            const vector_base<Space, RealType, F>& f,
            const vector_base<Space, RealType, G>& g) {
    auto fdata = transform_data(f.derived());
    auto gdata = transform_data(g.derived());
    constexpr std::size_t m = std::tuple_size<decltype(fdata)>::value;
    constexpr std::size_t n = std::tuple_size<decltype(gdata)>::value;
    return make_vector_transform<Space>(
      compose<m, n>(binary_op, transform_op(f.derived()), transform_op(g.derived())),
      std::tuple_cat(fdata, gdata)
    );
  }

} } // namespace libgm::experimental

#endif
