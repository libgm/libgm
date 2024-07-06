#pragma once

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace libgm {

/**
 * Returns the first k elements of a random permutation of 0,...,n-1.
 */
template <typename Generator>
uint_vector
randperm(Generator& rng, size_t n, size_t k) {
  if (k > n) {
    throw std::invalid_argument("randperm: k must be <= n");
  }
  uint_vector perm(n);
  std::iota(perm.begin(), perm.end(), size_t(0));
  for (size_t i = 0; i < k; ++i) {
    size_t j = std::uniform_int_distribution<size_t>(i, n - 1)(rng);
    std::swap(perm[i], perm[j]);
  }
  perm.resize(k);
  return perm;
}

/**
 * Returns a random permutation of 0,...,n-1.
 */
template <typename Generator>
uint_vector
randperm(Generator& rng, size_t n) {
  uint_vector perm(n);
  std::iota(perm.begin(), perm.end(), size_t(0));
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

/// @}

} // namespace libgm
