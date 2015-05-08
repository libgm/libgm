#ifndef LIBGM_RANDOM_PERMUTATIONS_HPP
#define LIBGM_RANDOM_PERMUTATIONS_HPP

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace libgm {

  /**
   * Returns the first k elements of a random permutation of 0,...,n-1.
   */
  template <typename Generator>
  std::vector<std::size_t>
  randperm(std::size_t n, Generator& rng, std::size_t k) {
    if (k > n) {
      throw std::invalid_argument("randperm: k must be <= n");
    }
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t(0));
    for (std::size_t i = 0; i < k; ++i) {
      std::size_t j = std::uniform_int_distribution<std::size_t>(i, n - 1)(rng);
      std::swap(perm[i], perm[j]);
    }
    perm.resize(k);
    return perm;
  }

  /**
   * Returns a random permutation of 0,...,n-1.
   */
  template <typename Generator>
  std::vector<std::size_t>
  randperm(std::size_t n, Generator& rng) {
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t(0));
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
  }

  //! @}

} // namespace libgm

#endif
