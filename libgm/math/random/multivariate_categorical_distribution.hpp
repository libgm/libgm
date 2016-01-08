#ifndef LIBGM_MULTIVARIATE_CATEGORICAL_DISTRIBUTION_HPP
#define LIBGM_MULTIVARIATE_CATEGORICAL_DISTRIBUTION_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/tags.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

  /**
   * A categorical distribution over multiple arguments,
   * whose probabilities are represented by a table.
   */
  template <typename T = double>
  class multivariate_categorical_distribution {
  public:

    //! The type representing the parameters of the distribution.
    typedef table<T> param_type;

    //! The type representing the sample.
    typedef uint_vector result_type;

    //! The type representing the assignment to the tail.
    typedef uint_vector tail_type;

    //! Constructor for a distribution in the probability space.
    explicit multivariate_categorical_distribution(const table<T>& p)
      : psum_(p) {
      std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
    }

    //! Constructor for a distribution in the log space.
    multivariate_categorical_distribution(const table<T>& lp, log_tag)
      : psum_(lp.shape()) {
      std::transform(lp.begin(), lp.end(), psum_.begin(), exponent<T>());
      std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
    }

    /**
     * Draws a random sample from a marginal distribution.
     * The distribution parameters must be normalized, so that the values
     * sum to 1.
     */
    template <typename Generator>
    uint_vector operator()(Generator& rng) const {
      return operator()(rng, uint_vector());
    }

    /**
     * Draws a random sample from a conditional distribution.
     * The distribution must be normalized, so that the all the values
     * for the given tail index sum to one.
     */
    template <typename Generator>
    uint_vector operator()(Generator& rng, const uint_vector& tail) const {
      assert(tail.size() < psum_.arity());
      std::size_t nhead = psum_.arity() - tail.size();
      std::size_t nelem = psum_.offset().multiplier(nhead);
      const T* begin = psum_.begin() + psum_.offset().linear(tail, nhead);
      T p = std::uniform_real_distribution<T>()(rng);
      if (begin > psum_.begin()) { p += *(begin-1); }
      std::size_t i = std::upper_bound(begin, begin + nelem, p) - begin;
      if (i < nelem) {
        return psum_.offset().finite(i, nhead);
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

  private:
    //! The table of partial sums of probabilities.
    table<T> psum_;

  }; // class multivariate_categorical_distribution

} // namespace libgm

#endif
