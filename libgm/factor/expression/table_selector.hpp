#ifndef LIBGM_TABLE_SELECTOR_HPP
#define LIBGM_TABLE_SELECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/factor/expression/table_base.hpp>
#include <libgm/factor/expression/table_function.hpp>
#include <libgm/factor/expression/table_join.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/tags.hpp>

#include <functional>
#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * The selector for probability_table expressions.
   *
   * \tparam IndexRange
   *         A range of dimensions to select, e.g., span or const uint_vector&.
   * \tparam F
   *         A probability_table expression (may be qualified as const).
   */
  template <typename Space, typename IndexRange, typename F>
  class table_selector {
  public:
    //! The real_type of the underlying expression.
    using real_type = typename std::remove_const_t<F>::real_type;

    //! The param_type of the underlying expression.
    using param_type = typename std::remove_const_t<F>::param_type;

    //! The multiplication operation.
    using multiplies_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      std::multiplies<real_type>,
      std::plus<real_type>
    >;

    //! The division operation.
    using divides_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      safe_divides<real_type>,
      std::minus<real_type>
    >;

    //! The summation operation.
    using plus_op = std::conditional_t<
      std::is_same<Space, prob_tag>::value,
      std::plus<real_type>,
      log_plus_exp<real_type>
    >;

    //! Constructs the selector with the given expression and range.
    table_selector(IndexRange dims, F& f)
      : dims_(dims), f_(f) { }

    /**
     * Multiplies a selector by a table expression with the same space, joining
     * the selected dimensions on the left and all the dimensions on the right.
     */
    template <typename Other>
    friend table_join<Space, multiplies_op, IndexRange, all, F, Other>
    operator*(const table_selector& f,
              const table_base<Space, real_type, Other>& g) {
      std::size_t arity = g.derived().arity();
      return { multiplies_op(), f.dims_, all(arity), f.f_, g.derived() };
    }

    /**
     * Multiplies a table expression by a selector with the same space, joining
     * all the dimensions on the left with selected dimensions on the right.
     */
    template <typename Other>
    friend table_join<Space, multiplies_op, all, IndexRange, Other, F>
    operator*(const table_base<Space, real_type, Other>& f,
              const table_selector& g) {
      std::size_t arity = f.derived().arity();
      return { multiplies_op(), all(arity), g.dims_, f.derived(), g.f_ };
    }

    /**
     * Multiplies two selectors with identical space, joining the selected
     * dimensions on the left with the selected dimensions on the right.
     */
    template <typename IndexRange2, typename G>
    friend table_join<Space, multiplies_op, IndexRange, IndexRange2, F, G>
    operator*(const table_selector& f,
              const table_selector<Space, IndexRange2, G>& g) {
      return { multiplies_op(), f.dims_, g.dims_, f.f_, g.f_ };
    }

    /**
     * Divides a selector by a table expression with the same space, joining
     * the selected dimensions on the left and all the dimensions on the right.
     */
    template <typename Other>
    friend table_join<Space, divides_op, IndexRange, all, F, Other>
    operator/(const table_selector& f,
              const table_base<Space, real_type, Other>& g) {
      std::size_t arity = g.derived().arity();
      return { divides_op(), f.dims_, all(arity), f.f_, g.derived() };
    }

    /**
     * Divides a table expression by a selector with the same space, joining
     * all the dimensions on the left with selected dimensions on the right.
     */
    template <typename Other>
    friend table_join<Space, divides_op, all, IndexRange, Other, F>
    operator/(const table_base<Space, real_type, Other>& f,
              const table_selector& g) {
      std::size_t arity = f.derived().arity();
      return { divides_op(), all(arity), g.dims_, f.derived(), g.f_ };
    }

    /**
     * Divides two selectors with identical space, joining the selected
     * dimensions on the left with the selected dimensions on the right.
     */
    template <typename IndexRange2, typename G>
    friend table_join<Space, divides_op, IndexRange, IndexRange2, F, G>
    operator/(const table_selector& f,
              const table_selector<Space, IndexRange2, G>& g) {
      return { divides_op(), f.dims_, g.dims_, f.f_, g.f_ };
    }

    /**
     * Returns a table expression that eliminates the selected dimensions
     * using the specified binary operation and initial value.
     */
    template <typename AggOp>
    auto eliminate(AggOp agg_op, real_type init) const {
      return make_table_function<Space>(
        [agg_op, init, dims = dims_](const F& f, param_type& result) {
          f.param().aggregate(agg_op, init, dims.complement(f.arity()), result);
        }, f_.arity() - dims_.size(), f_
      );
    }

    /**
     * Returns a table expression with space given by Space, summing out the
     * selected dimensions of the underlying expression.
     */
    auto sum() const {
      return eliminate(plus_op(), real_type(0));
    }

    /**
     * Returns a table expression with space given by Space, maximizing out the
     * selected dimensions of the underlying expression.
     */
    auto max() const {
      return eliminate(maximum<real_type>(), -inf<real_type>());
    }

    /**
     * Returns a table expression with space given by Space, minimizing out the
     * selected dimensions of the underlying expression.
     */
    auto min() const {
      return eliminate(minimum<real_type>(), +inf<real_type>());
    }

    /**
     * Multiplies another expression into the underlying expression.
     * Only supported when the expression is mutable.
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    table_selector& operator*=(const table_base<Space, real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      g.derived().join_inplace(multiplies_op(), dims_, f_.param());
      return *this;
    }

    /**
     * Divides another expression into the underlying expression.
     * Only supported when the expression is mutable.
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    table_selector& operator/=(const table_base<Space, real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      g.derived().join_inplace(divides_op(), dims_, f_.param());
      return *this;
    }

  private:
    IndexRange dims_;
    add_reference_if_factor_t<F> f_;

    template <typename OtherSpace, typename IndexRange2, typename G>
    friend class table_selector;

  }; // class table_selector

} } // namespace libgm::experimental

#endif
