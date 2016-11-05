#ifndef LIBGM_CANONICAL_GAUSSIAN_SELECTOR_HPP
#define LIBGM_CANONICAL_GAUSSIAN_SELECTOR_HPP

#include <libgm/enable_if.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_base.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_function.hpp>
#include <libgm/factor/experimental/expression/canonical_gaussian_join.hpp>
#include <libgm/factor/traits.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/math/param/canonical_gaussian_param.hpp>

#include <type_traits>

namespace libgm { namespace experimental {

  /**
   * A canonical_gaussian selector that references a range of dimensions
   * of an underlying canonical_gaussian expression.
   */
  template <typename IndexRange, typename F>
  class canonical_gaussian_selector {
  public:
    // shortcuts
    using real_type  = typename std::remove_const_t<F>::real_type;
    using param_type = canonical_gaussian_param<real_type>;

    canonical_gaussian_selector(IndexRange dims, F& f)
      : dims_(dims), f_(f) { }

    /**
     * Returns a canonical_gaussian expression representing the product
     * of a canonical_gaussian selector and expression.
     */
    template <typename Other>
    friend canonical_gaussian_join<std::plus<>, IndexRange, all, F, Other>
    operator*(const canonical_gaussian_selector& f,
              const canonical_gaussian_base<real_type, Other>& g) {
      std::size_t arity = g.derived().arity();
      return { std::plus<>(), f.dims_, all(arity), f.f_, g.derived() };
    }

    /**
     * Returns a canonical_gaussian expression representing the product
     * of a canonical_gaussian expression and selector.
     */
    template <typename Other>
    friend canonical_gaussian_join<std::plus<>, all, IndexRange, Other, F>
    operator*(const canonical_gaussian_base<real_type, Other>& f,
              const canonical_gaussian_selector& g) {
      std::size_t arity = f.derived().arity();
      return { std::plus<>(), all(arity), g.dims_, f.derived(), g.f_ };
    }

    /**
     * Returns a canonical_gaussian expression representing the product
     * of two canonical_gaussian selectors.
     */
    template <typename IndexRange2, typename G>
    friend canonical_gaussian_join<std::plus<>, IndexRange, IndexRange2, F, G>
    operator*(const canonical_gaussian_selector& f,
              const canonical_gaussian_selector<IndexRange2, G>& g) {
      return { std::plus<>(), f.dims_, g.dims_, f.f_, g.f_ };
    }

    /**
     * Returns a canonical_gaussian expression representing the ratio
     * of a canonical_gaussian selector and expression.
     */
    template <typename Other>
    friend canonical_gaussian_join<std::minus<>, IndexRange, all, F, Other>
    operator/(const canonical_gaussian_selector& f,
              const canonical_gaussian_base<real_type, Other>& g) {
      std::size_t arity = g.derived().arity();
      return { std::minus<>(), f.dims_, all(arity), f.f_, g.derived() };
    }

    /**
     * Returns a canonical_gaussian expression representing the ratio
     * of a canonical_gaussian expression and selector.
     */
    template <typename Other>
    friend canonical_gaussian_join<std::minus<>, all, IndexRange, Other, F>
    operator/(const canonical_gaussian_base<real_type, Other>& f,
              const canonical_gaussian_selector& g) {
      std::size_t arity = f.derived().arity();
      return { std::minus<>(), all(arity), g.dims_, f.derived(), g.f_ };
    }

    /**
     * Returns a canonical_gaussian expression representing the ratio
     * of two canonical_gaussian selectors.
     */
    template <typename IndexRange2, typename G>
    friend canonical_gaussian_join<std::minus<>, IndexRange, IndexRange2, F, G>
    operator/(const canonical_gaussian_selector& f,
              const canonical_gaussian_selector<IndexRange2, G>& g) {
      return { std::minus<>(), f.dims_, g.dims_, f.f_, g.f_ };
    }

    /**
     * Returns a canonical_gaussian expression that represents elimination
     * (sum or maximum) of a canonical_gaussian over the selected dimensions.
     */
    auto eliminate(bool marginal) const {
      using workspace_type = typename param_type::collapse_workspace;
      return make_canonical_gaussian_function<workspace_type>(
        [marginal, dims = dims_](const F& f, workspace_type& ws,
                                 param_type& result) {
          f.param().collapse(marginal, dims.complement(f.arity()), dims, ws,
                             result);
        }, f_.arity() - dims_.size(), f_);
    }

    /**
     * Returns a canonical_gaussian expression representing the sum
     * of this expression over the selected dimensions.
     */
    auto sum() const {
      return eliminate(true /* marginal */);
    }

    /**
     * Returns a canonical_gaussian expression representing the maximum
     * of this expression over the selected dimensions.
     */
    auto max() const {
      return eliminate(false /* maximum */);
    }

    /**
     * Multiplies a canonical_gaussian expression into the underlying factor.
     * Only supported when the expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    canonical_gaussian_selector&
    operator*=(const canonical_gaussian_base<real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      g.derived().join_inplace(plus_assign<>(), dims_, f_.param());
      return *this;
    }

    /**
     * Divides a canonical_gaussian expression into the underlying factor.
     * Only supported when this expression is mutable (e.g., a factor).
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    canonical_gaussian_selector&
    operator/=(const canonical_gaussian_base<real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      g.derived().join_inplace(minus_assign<>(), dims_, f_.param());
      return *this;
    }

  private:
    IndexRange dims_;
    add_reference_if_factor_t<F> f_;

    template <typename IndexRange2, typename G>
    friend class canonical_gaussian_selector;

  }; // class canonical_gaussian_selector

} } // namespace libgm::experimental

#endif
