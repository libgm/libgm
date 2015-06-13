#ifndef LIBGM_SOFTMAX_DISTRIBUTION_HPP
#define LIBGM_SOFTMAX_DISTRIBUTION_HPP

#include <libgm/math/param/softmax_param.hpp>

#include <random>

namespace libgm {

  /**
   * A class that can draw random samples from the conditional distribution
   * given by the softmax
   * p(y=i | x) = exp(b_i + w_i^T x) / sum_j exp(b_j + w_j^T x).
   */
  template <typename T = double>
  class softmax_distribution {
  public:
    //! The type representing the parameters of the distribution
    typedef softmax_param<T> param_type;

    //! The type representing a sample
    typedef std::size_t result_type;

    //! The type representing the assignment to the tail.
    typedef real_vector<T> tail_type;

    //! Constructs a soft max distribution with the given parameters.
    softmax_distribution(const softmax_param<T>& param)
      : param_(param) { }

    //! Returns the parameters of this distribution
    const param_type& param() const {
      return param_;
    }

    //! Draws a random sample from a softmax distribution
    template <typename Generator>
    std::size_t operator()(Generator& rng, const tail_type& tail) const {
      return param_.sample(rng, tail);
    }

  private:
    //! The parameters determining the softmax distribution.
    softmax_param<T> param_;

  }; // class softmax_distribution

} // namespace libgm

#endif
