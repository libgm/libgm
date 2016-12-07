#ifndef LIBGM_MATRIX_SELECTOR_HPP
#define LIBGM_MATRIX_SELECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/functional/member.hpp>
#include <libgm/factor/expression/matrix_base.hpp>
#include <libgm/factor/expression/matrix_function.hpp>
#include <libgm/factor/expression/matrix_vector_join.hpp>
#include <libgm/factor/expression/vector_matrix_join.hpp>
#include <libgm/factor/utility/traits.hpp>
#include <libgm/functional/updater.hpp>
#include <libgm/math/eigen/join.hpp>
#include <libgm/math/tags.hpp>
#include <libgm/traits/int_constant.hpp>

#include <functional>
#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * A selector that references a dimension (rows or columns) of a matrix
   * expression.
   *
   * \tparam Direction
   *         one of Eigen::Vertical, Eigen::Horizontal, or Eigen::BothDirections
   */
  template <typename Space, int Direction, typename F>
  class matrix_selector {
  public:
    // shortcuts
    using real_type = typename std::remove_const_t<F>::real_type;

    //! The multiplication operation.
    using multiplies_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      std::multiplies<>,
      std::plus<>
    >;

    //! The division operation.
    using divides_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      std::divides<>,
      std::minus<>
    >;

    //! The summation operation.
    using sum_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      member_sum,
      member_logSumExpVectorwise
    >;

    /**
     * Constructs a selector with fixed dimension.
     */
    LIBGM_ENABLE_IF(Direction != Eigen::BothDirections)
    matrix_selector(F& f)
      : f_(f) { }

    /**
     * Constructs a selector referencing a dimensions of f at runtime.
     */
    LIBGM_ENABLE_IF(Direction == Eigen::BothDirections)
    matrix_selector(std::size_t dim, F& f)
      : dim_(dim), f_(f) { }

    /**
     * Returns a matrix expression representing the product of a matrix selector
     * and a vector expression with identical space and real type.
     */
    template <typename Other>
    friend matrix_vector_join<Space, multiplies_op, Direction, F, Other>
    operator*(const matrix_selector& f,
              const vector_base<Space, real_type, Other>& g) {
      return { multiplies_op(), f.dim_, f.f_, g.derived() };
    }

    /**
     * Returns a matrix expression representing the product of a vector
     * expression and a matrix selector with identical space and real type.
     */
    template <typename Other>
    friend vector_matrix_join<Space, multiplies_op, Direction, Other, F>
    operator*(const vector_base<Space, real_type, Other>& f,
              const matrix_selector& g) {
      return { multiplies_op(), g.dim_, f.derived(), g.f_ };
    }

    /**
     * Returns a matrix expression representing the division of a matrix
     * selector and a vector expression with identical space and real type.
     */
    template <typename Other>
    friend matrix_vector_join<Space, divides_op, Direction, F, Other>
    operator/(const matrix_selector& f,
              const vector_base<Space, real_type, Other>& g) {
      return { divides_op(), f.dim_, f.f_, g.derived() };
    }

    /**
     * Returns a matrix expression representing the division of a vector
     * expression and a matrix selector with identical space and real type.
     */
    template <typename Other>
    friend vector_matrix_join<Space, divides_op, Direction, Other, F>
    operator/(const vector_base<Space, real_type, Other>& f,
              const matrix_selector& g) {
      return { divides_op(), g.dim_, f.derived(), g.f_ };
    }

    /**
     * Returns a vector expression with space given by Space, summing out the
     * selected dimenion of the underlying matrix expression.
     */
    auto sum() const {
      return eliminate(sum_op(), int_constant<Direction>());
    }

    /**
     * Returns a vector expression with space given by Space, maximizing out the
     * selected dimension of the underlying matrix expression.
     */
    auto max() const {
      return eliminate(member_maxCoeff(), int_constant<Direction>());
    }

    /**
     * Returns a vector expression with space given by Space, minimizing out the
     * selected dimension of the underlying matrix expression.
     */
    auto min() const {
      return eliminate(member_minCoeff(), int_constant<Direction>());
    }

    /**
     * Multiplies a vector expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    matrix_selector& operator*=(const vector_base<Space, real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      update(multiplies_op(), g);
      return *this;
    }

    /**
     * Divides a probability_vector expression into this expression.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    matrix_selector& operator/=(const vector_base<Space, real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      update(divides_op(), g);
      return *this;
    }

  private:
    template <typename AggOp>
    auto eliminate(AggOp agg_op, int_constant<Eigen::Vertical>) const {
      return make_vector_function<Space>(
        [agg_op](const F& f, dense_vector<real_type>& result) {
          result = agg_op(f.param().colwise()).transpose();
        }, f_);
    }

    template <typename AggOp>
    auto eliminate(AggOp agg_op, int_constant<Eigen::Horizontal>) const {
      return make_vector_function<Space>(
        [agg_op](const F& f, dense_vector<real_type>& result) {
          result = agg_op(f.param().rowwise());
        }, f_);
    }

    template <typename AggOp>
    auto eliminate(AggOp agg_op, int_constant<Eigen::BothDirections>) const {
      return make_vector_function<Space>(
        [agg_op, dim = dim_](const F& f, dense_vector<real_type>& result) {
          if (dim == 0) {
            result = agg_op(f.param().colwise()).transpose();
          } else {
            result = agg_op(f.param().rowwise());
          }
        }, f_);
    }

    template <typename JoinOp, typename Other>
    void update(JoinOp op, const vector_base<Space, real_type, Other>& f) const {
      if (f.derived().alias(f_.param())) {
        mv_join(make_updater(op), f_.param(), f.derived().param().eval(), dim_,
                int_constant<Direction>());
      } else {
        mv_join(make_updater(op), f_.param(), f.derived().param(), dim_,
                int_constant<Direction>());
      }
    }

    std::size_t dim_;
    add_reference_if_factor_t<F> f_;

  }; // class matrix_selector

} } // namespace libgm::experimental

#endif
