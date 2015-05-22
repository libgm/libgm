#ifndef LIBGM_MULTINOMIAL_DISTRIBUTION_HPP
#define LIBGM_MULTINOMIAL_DISTRIBUTION_HPP

#include <libgm/math/numerical_error.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace libgm {

  /**
   * A class that represents a multinomial distribution:
   * \f$p(x = i) = p_i\f$ for \f$i = 0, \ldots, n-1\f$.
   *
   * \tparam T a real type representing the parameters
   */
  template <typename T = double>
  class multinomial_distribution {
  public:
    //! The type of outcomes.
    typedef std::size_t result_type;

    //! The type representing the parameters.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> param_type;

    //! Constructs a multinomial distribution with the given parameters.
    explicit multinomial_distribution(const param_type& param)
      : psum_(param.size()) {
      std::partial_sum(param.data(), param.data() + param.size(), psum_.data());
    }

    /**
     * Draws a sample from the distribution for a single trial using
     * the specified random number generator.
     */
    template <typename Generator>
    std::size_t operator()(Generator& rng) const {
      std::uniform_real_distribution<T> unif;
      auto begin = psum_.data(), end = psum_.data() + psum_.size();
      auto it = std::upper_bound(begin, end, unif(rng));
      if (it == end) {
        throw numerical_error("The probabilities are less than 1");
      } else {
        return it - begin;
      }
    }

    /**
     * Draws samples from the distribution for multiple trials using
     * the specified random number generator.
     */
    template <typename Generator>
    std::vector<std::size_t> operator()(std::size_t n, Generator& rng) const {
      std::vector<std::size_t> result(n);
      std::generate_n(result.begin(), n, std::bind(this, std::ref(rng)));
      return result;
    }

  private:
    //! The vector of partial sums.
    param_type psum_;

  }; // class multinomial_distribution

} // namespace libgm

#endif
