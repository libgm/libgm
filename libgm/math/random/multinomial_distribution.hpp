#ifndef LIBGM_MULTINOMIAL_DISTRIBUTION_HPP
#define LIBGM_MULTINOMIAL_DISTRIBUTION_HPP

#include <libgm/math/numerical_error.hpp>

#include <numeric>
#include <random>
#include <vector>

namespace Eigen {
  // Forward declaration, so that we do not need to include Eigen by default
  template <typename Derived> class DenseBase;
}

namespace libgm {

  /**
   * A class that represents a multinomial distribution:
   * \f$p(x = i) = p_i\f$ for \f$i = 0, \ldots, n-1\f$.
   *
   * \tparam T a real type representing the parameters
   */
  template <typename T>
  class multinomial_distribution {
  public:
    //! The type of outcomes.
    typedef std::size_t result_type;

    //! The type representing the parameters.
    typedef std::vector<T> param_type;

    //! Constructs a multinomial distribution with the given parameters.
    explicit multinomial_distribution(const param_type& param)
      : psum_(param) {
      std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
    }

    /**
     * Constructs a multinomial distribution with parameters specified
     * by a 1D Eigen vector / array.
     */
    template <typename Derived>
    explicit multinomial_distribution(const Eigen::DenseBase<Derived>& derived)
      : psum_(derived.size()) {
      T accu(0);
      for (std::size_t i = 0; i < psum_.size(); ++i) {
        accu += derived[i];
        psum_[i] = accu;
      }
    }

    //! Draws a sample from the distribution using the specified generator.
    template <typename Generator>
    std::size_t operator()(Generator& rng) const {
      std::uniform_real_distribution<T> unif;
      auto it = std::upper_bound(psum_.begin(), psum_.end(), unif(rng));
      if (it == psum_.end()) {
        throw numerical_error("The probabilities are less than 1");
      } else {
        return it - psum_.begin();
      }
    }
  private:
    //! The vector of partial sums.
    param_type psum_;

  }; // class multinomial_distribution

} // namespace libgm

#endif
