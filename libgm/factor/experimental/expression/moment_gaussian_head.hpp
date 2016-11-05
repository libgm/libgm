#ifndef LIBGM_MOMENT_GAUSSIAN_HEAD_HPP
#define LIBGM_MOMENT_GAUSSIAN_HEAD_HPP

#include <libgm/factor/experimental/expression/moment_gaussian_base.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_function.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm { namespace experimental {

  /**
   * A moment_gaussian selector that references a range of head
   * dimensions of an underlying moment_gaussian expression.
   */
  template <typename IndexRange, typename F>
  class moment_gaussian_head {
  public:
    // Shortcuts
    using real_type  = typename std::remove_const_t<F>::real_type;
    using param_type = moment_gaussian_param<real_type>;

    /**
     * Constructs the selector.
     */
    moment_gaussian_head(IndexRange dims, F& f)
      : dims_(dims), f_(f) { }

    /**
     * Returns a moment_gaussian expression that eliminates the selected
     * dimensions.
     */
    auto eliminate(bool marginal) const {
      return make_moment_gaussian_function<void>(
        [marginal, dims = dims_](const F& f, param_type& result) {
          f.param().collapse(marginal, complement(dims, f.head_arity()), result);
        }, f_.head_arity() - dims_.size(), f_.tail_arity(), f_);
    }

    /**
     * Returns a moment_gaussian expression representing the sum
     * of this expression over the selected dimensions.
     */
    auto sum() const {
      return eliminate(true /* marginal */);
    }

    /**
     * Returns a moment_guassian expression representing the maximum
     * of this expression over the seleced dimensions.
     */
    auto max() const {
      return eliminate(false /* maximum */);
    }

    /**
     * Multiplies another expression into this expression.
     * Only supported when this expression is mutable (eg., a factor).
     */
    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    moment_gaussian_head&
    operator*=(const moment_gaussian_base<real_type, Other>& g) {
      assert(!g.derived().alias(f_.param()));
      g.derived().multiply_inplace(dims_, f_.param());
      return *this;
    }

  private:
    IndexRange dims_;
    add_reference_if_factor_t<F> f_;

    template <typename IndexRange2, typename G>
    friend class moment_gaussian_tail;

  }; // class moment_gaussian_head

} } // namespace libgm::experimental

#endif
