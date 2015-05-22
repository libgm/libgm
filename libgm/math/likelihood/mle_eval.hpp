#ifndef LIBGM_MLE_EVAL_HPP
#define LIBGM_MLE_EVAL_HPP

namespace libgm {

  /**
   * A function that implements operator() for a class that models the
   * IncrementalMLE concept.
   */
  template <typename Mle, typename Range, typename Shape>
  typename Mle::param_type
  incremental_mle_eval(Mle& mle, const Range& data, const Shape& shape) {
    mle.initialize(shape);
    for (const auto& d : data) {
      mle.process(d.first, d.second);
    }
    return mle.param();
  }

  /**
   * A function that implements operator() for a class that models the
   * IterativeMLE concept.
   */
  template <typename Mle, typename Range, typename Shape>
  typename Mle::param_type
  iterative_mle_eval(Mle& mle, const Range& data, const Shape& shape,
                     typename Mle::real_type tol, std::size_t max_iter) {
    mle.initialize(shape);
    for (std::size_t it = 0; it < max_iter; ++it) {
      for (const auto& d : data) {
        mle.process(d.first, d.second);
      }
      if (mle.update() < tol) {
        break;
      }
    }
    return mle.param();
  }

} // namespace libgm

#endif

