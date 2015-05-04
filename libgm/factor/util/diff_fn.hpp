#ifndef LIBGM_FACTOR_DIFF_FN_HPP
#define LIBGM_FACTOR_DIFF_FN_HPP

#include <functional>

namespace libgm {

  //! \addtogroup factor_types
  //! @{

  /**
   * A typedef for a function computing the difference between the
   * parameters of two factors.
   */
  template <typename F>
  using diff_fn = std::function<typename F::real_type(const F&, const F&)>;

  /**
   * Returns an object that computes the sum-of-absolute-differences
   * between two factors.
   * \tparam F A factor type that supports the sum_diff free function.
   */
  template <typename F>
  diff_fn<F> sum_diff_fn() {
    return [](const F& f, const F& g) { return sum_diff(f, g); };
  }

  /**
   * Returns an object that computes the maximum-of-absolute differences
   * between two factors.
   * \tparam F A factor type that supports the max_diff free function.
   */
  template <typename F>
  diff_fn<F> max_diff_fn() {
    return [](const F& f, const F& g) { return max_diff(f, g); };
  }

  //! @} group factor_types
}

#endif
