#ifndef LIBGM_BIVARIATE_CATEGORICAL_DISTRIBUTION_HPP
#define LIBGM_BIVARIATE_CATEGORICAL_DISTRIBUTION_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/tags.hpp>

#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

  /**
   * A categorical distribution over two variables,
   * whose parameters are represented by a dense Eigen matrix.
   */
  template <typename T = double>
  class bivariate_categorical_distribution {
  public:
    //! The underlying parameter type.
    typedef real_matrix<T> param_type;

    //! The type representing the sample.
    typedef std::pair<std::size_t, std::size_t> result_type;

    //! The type representing the assignment to the tail.
    typedef std::size_t tail_type;

    //! Constructor for a distribution in the probability space.
    bivariate_categorical_distribution(const real_matrix<T>& p, prob_tag)
      : psum_(p) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Constructor for a distribution in the log space.
    bivariate_categorical_distribution(const real_matrix<T>& p, log_tag)
      : psum_(exp(p.array())) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    std::pair<std::size_t, std::size_t> operator()(Generator& rng) const {
      const T* begin = psum_.data();
      T p = std::uniform_real_distribution<T>()(rng);
      std::ptrdiff_t i =
        std::upper_bound(begin, begin + psum_.size(), p) - begin;
      if (i < psum_.size()) {
        return { i % psum_.rows(), i / psum_.rows() };
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    std::size_t operator()(Generator& rng, std::size_t tail) const {
      const T* begin = psum_.data() + tail * psum_.rows();
      T p = std::uniform_real_distribution<T>()(rng);
      if (tail > 0) { p += *(begin-1); }
      std::ptrdiff_t i =
        std::upper_bound(begin, begin + psum_.rows(), p) - begin;
      if (i < psum_.rows()) {
        return i;
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

  private:
    //! Partial sums.
    real_matrix<T> psum_;

  }; // class bivariate_categorical_distribution.

} // namespace libgm

#endif
