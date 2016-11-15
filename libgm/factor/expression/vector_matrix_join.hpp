#ifndef LIBGM_VECTOR_MATRIX_JOIN_HPP
#define LIBGM_VECTOR_MATRIX_JOIN_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/expression/matrix_base.hpp>
#include <libgm/factor/expression/vector_function.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/compose.hpp>
#include <libgm/math/eigen/join.hpp>
#include <libgm/math/eigen/multiply.hpp>
#include <libgm/traits/int_constant.hpp>

#include <functional>

namespace libgm { namespace experimental {

  /**
   * A class that represents a binary join of a vector and a matrix along
   * the dimension specified via the Direction template parameter.
   *
   * The Direction argument must take on one of the values specified by Eigen's
   * DirectionType enum (see Eigen/src/Core/util/Constants.h).
   * When Direction = Eigen::Vertical or Eigen::Horizontal, the joined dimension
   * is selected at compile-time (column-wise or row-wise, respectively).
   * When Direction = Eigen::BothDirections, the joined dimension is selected
   * at run-time, based on the constructor argument.
   */
  template <typename Space, typename JoinOp, int Direction,
            typename F, typename G>
  class vector_matrix_join
    : public matrix_base<
        Space,
        typename F::real_type,
        vector_matrix_join<Space, JoinOp, Direction, F, G> > {
    static_assert(
      std::is_same<typename F::real_type, typename G::real_type>::value,
      "The joined expressions must have the same real type");
    static_assert(
      is_vector<F>::value && is_matrix<G>::value,
      "This expression must join a vector and a matrix");

  public:
    using real_type  = typename F::real_type;
    using param_type = dense_matrix<real_type>;
    using base = matrix_base<Space, real_type, vector_matrix_join>;

    using base::aggregate;

    vector_matrix_join(JoinOp join_op, std::size_t dim, const F& f, const G& g)
      : join_op_(join_op), dim_(dim), f_(f), g_(g) {
      assert(Direction != Eigen::BothDirections || dim <= 1);
    }

    bool alias(const dense_vector<real_type>& param) const {
      return false;
      // matrix_vector_join::param() returns a temporary, so it cannot alias
      // a vector
    }

    bool alias(const dense_matrix<real_type>& param) const {
      return f_.alias(param) || g_.alias(param);
      // f_ might alias a matrix e.g. if it is a segment of that matrix
      // g_ might alias a matrix e.g. if &param == &g_.param()
    }

    void eval_to(param_type& result) const {
      vm_join([join_op = join_op_, &result](const auto& vec, const auto& mat) {
          result = join_op(vec, mat);
        }, f_.param(), g_.param(), dim_, int_constant<Direction>());
    }

    template <typename ResultSpace = Space, typename UnaryOp = void>
    matrix_vector_join<ResultSpace, compose_t<UnaryOp, JoinOp>, Direction, F, G>
    transform(UnaryOp unary_op) const {
      return { compose(unary_op, join_op_), dim_, f_, g_ };
    }

    LIBGM_ENABLE_IF((std::is_same<JoinOp, std::multiplies<> >::value))
    auto aggregate(member_sum, std::size_t retain) const {
      // for now, disallow meaningless join-aggregate
      assert(retain != (Direction == Eigen::BothDirections ? dim_ : Direction));
      return make_vector_function<Space>(
        [dim = dim_](const F& f, const G& g, dense_vector<real_type>& result) {
          mv_multiply(g.param(), f.param(), result, dim,
                      int_constant<Direction>());
        }, f_, g_);
    }

  private:
    JoinOp join_op_;
    std::size_t dim_;
    add_const_reference_if_factor_t<F> f_;
    add_const_reference_if_factor_t<G> g_;

  }; // class vector_matrix_join

} } // namespace libgm::experimental

#endif
