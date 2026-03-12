#pragma once

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace libgm {

/**
 * A mixture distribution.
 * \tparam Base a distribution of each component
 */
template <typename F>
class MixtureDistribution {
public:
  //! The type representing the sample.
  using result_type = typename F::distribution_type::result_type;

  //! The type representing real values.
  using real_type = typename F::real_type;

  /**
   * Constructs a marginal distribution with the given parameters.
   */
  explicit MixtureDistribution(const Mixture<F>& mixture) {
    base_.reserve(mixture.size());
    psum_.reserve(mixture.size());
    for (const auto& p : param) {
      base_.emplace_back(p);
      psum_.emplace_back(p.marginal());
    }
    std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
  }

  /**
   * Draws a random sample from a marginal distribution.
   */
  template <typename Generator>
  result_type operator()(Generator& rng) const {
    real_type p = std::uniform_real_distribution<real_type>()(rng);
    size_t i =
      std::upper_bound(psum_.begin(), psum_.end(), p) - psum_.begin();
    if (i < psum_.size()) {
      return base_[i](rng);
    } else {
      throw std::invalid_argument("The total probability is less than 1");
    }
  }

private:
  //! The distribution for each component.
  std::vector<typename F::distribution_type> base_;

  //! The partial sums of component weights (the last one should be 1).
  std::vector<real_type> psum_;

};

}
