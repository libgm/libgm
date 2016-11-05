#ifndef LIBGM_CATEGORICAL_DISTRIBUTION_HPP
#define LIBGM_CATEGORICAL_DISTRIBUTION_HPP

#include <libgm/math/eigen/real.hpp>
#include <libgm/math/tags.hpp>

#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

  /**
   * A categorical distribution over a single variable,
   * whose parameters are represented by a dense Eigen vector.
   */
  template <typename T = double>
  class categorical_distribution {
  public:
    //! The underlying parameter type.
    typedef real_vector<T> param_type;

    //! The type representing the sample.
    typedef std::size_t result_type;

    //! Constructor for a distribution in the probability space.
    categorical_distribution(const real_vector<T>& p, prob_tag)
      : psum_(p) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Constructor for a distribution in the log space.
    categorical_distribution(const real_vector<T>& p, log_tag)
      : psum_(exp(p.array())) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    std::size_t operator()(Generator& rng) const {
      const T* begin = psum_.data();
      T p = std::uniform_real_distribution<T>()(rng);
      std::ptrdiff_t i
        = std::upper_bound(begin, begin + psum_.size(), p) - begin;
      if (i < psum_.size()) {
        return i;
      } else {
        throw std::invalid_argument("The probabilities are less than 1");
      }
    }

  private:
    //! Partial sums.
    real_vector<T> psum_;

  }; // class categorical_distribution

} // namespace libgm

#endif
