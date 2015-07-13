#ifndef LIBGM_MIXTURE_DISTRIBUTION_HPP
#define LIBGM_MIXTURE_DISTRIBUTION_HPP

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace libgm {

  /**
   * A mixture distribution.
   * \tparam Base a distribution of each component
   */
  template <typename Base>
  class mixture_distribution {
  public:
    //! The type of parameters of this distribution.
    typedef std::vector<typename Base::param_type> param_type;

    //! The type representing the sample.
    typedef typename Base::result_type result_type;

    //! The type representing real values.
    typedef typename Base::param_type::value_type real_type;

    /**
     * Constructs a marginal distribution with the given parameters.
     */
    explicit mixture_distribution(const param_type& param) {
      base_.reserve(param.size());
      psum_.reserve(param.size());
      for (const auto& p : param) {
        base_.emplace_back(p);
        psum_.emplace_back(p.weight());
      }
      std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
    }

    /**
     * Draws a random sample from a marginal distribution.
     */
    template <typename Generator>
    result_type operator()(Generator& rng) const {
      real_type p = std::uniform_real_distribution()(rng);
      std::size_t i =
        std::upper_bound(psum_.begin(), psum_.end(), p) - psum_.begin();
      if (i < psum_.size()) {
        return base_[i](rng);
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

  private:
    //! The distribution for each component.
    std::vector<Base> base_;

    //! The partial sums of component weights (the last one should be 1).
    std::vector<real_type> psum_;

  }; // class mixture_distribution

} // namespace libgm

#endif
