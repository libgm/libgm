#ifndef LIBGM_MOMENT_GAUSSIAN_TAIL_HPP
#define LIBGM_MOMENT_GAUSSIAN_TAIL_HPP

#include <libgm/factor/experimental/expression/moment_gaussian_base.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_function.hpp>
#include <libgm/factor/experimental/expression/moment_gaussian_head.hpp>
#include <libgm/math/param/moment_gaussian_param.hpp>

namespace libgm { namespace experimental {

  /**
   * A moment_gaussian selector that references a range of tail
   * dimensions of an underlying moment_gaussian expression.
   */
  template <typename IndexRange, typename F>
  class moment_gaussian_tail {
  public:
    // Shortcuts
    using real_type  = typename std::remove_const_t<F>::real_type;
    using param_type = moment_gaussian_param<real_type>;

    /**
     * Constructs the selector.
     */
    moment_gaussian_tail(IndexRange dims, F& f)
      : dims_(dims), f_(f) { }

    template <typename HeadDims, typename Other>
    friend auto
    operator*(const moment_gaussian_head<HeadDims, Other>& f,
              const moment_gaussian_tail& g) {
      assert(f.dims_.size() == g.dims_.size());
      return make_moment_gaussian_function<void>(
        [fdims = f.dims_, gdims = g.dims_]
        (const Other& f, const F& g, param_type& result) {
          multiply_head_tail(f.param(), g.param(), fdims, gdims, true, result);
        },
        f.f_.head_arity() + g.f_.head_arity(),
        f.f_.tail_arity() + g.f_.tail_arity() - f.dims_.size(),
        f.f_, g.f_);
    }

    template <typename HeadDims, typename Other>
    friend auto
    operator*(const moment_gaussian_tail& f,
              const moment_gaussian_head<HeadDims, Other>& g) {
      assert(f.dims_.size() == g.dims_.size());
      return make_moment_gaussian_function<void>(
        [fdims = f.dims_, gdims = g.dims_]
        (const F& f, const Other& g, param_type& result) {
          multiply_head_tail(g.param(), f.param(), gdims, fdims, false, result);
        },
        f.f_.head_arity() + g.f_.head_arity(),
        f.f_.tail_arity() + g.f_.tail_arity() - f.dims_.size(),
        f.f_, g.f_);
    }

    template <typename TailDims, typename Other>
    friend auto
    operator*(const moment_gaussian_tail& f,
              const moment_gaussian_tail<TailDims, Other>& g) {
      assert(f.dims_.size() == g.dims_.size());
      return make_moment_gaussian_function<void>(
        [fdims = f.dims_, gdims = g.dims_]
        (const F& f, const Other& g, param_type& result) {
          multiply_tails(f.param(), g.param(), fdims, gdims, result);
        },
        f.f_.head_arity() + g.f_.head_arity(),
        f.f_.tail_arity() + g.f_.tail_arity() - f.dims_.size(),
        f.f_, g.f_);
    }

    LIBGM_ENABLE_IF_N(!std::is_const<F>::value, typename Other)
    moment_gaussian_tail&
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

  }; // class moment_gaussian_tail

} } // namespace libgm::experimental

#endif
