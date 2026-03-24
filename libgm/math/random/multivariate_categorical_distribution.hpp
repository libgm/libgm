#pragma once

#include <libgm/datastructure/table.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/arithmetic.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

/**
 * A categorical distribution over multiple arguments,
 * whose probabilities are represented by a Table.
 */
template <typename T = double>
class MultivariateCategoricalDistribution {
public:
  /// The type representing the sample.
  using result_type = std::vector<size_t>;

  /// The type representing the assignment to the tail.
  using tail_type = std::vector<size_t>;

  /// Constructor for a distribution in the probability space.
  explicit MultivariateCategoricalDistribution(const ProbabilityTable<T>& p)
    : psum_(p.param()) {
    std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
  }

  /// Constructor for a distribution in the log space.
  explicit MultivariateCategoricalDistribution(const LogarithmicTable<T>& lp)
    : psum_(lp.shape()) {
    std::transform(lp.begin(), lp.end(), psum_.begin(), ExponentOp<T>());
    std::partial_sum(psum_.begin(), psum_.end(), psum_.begin());
  }

  /**
   * Draws a random sample from a marginal distribution.
   * The distribution parameters must be normalized so that the values sum to 1.
   */
  template <typename Generator>
  std::vector<size_t> operator()(Generator& rng) const {
    return operator()(rng, {});
  }

  /**
   * Draws a random sample from a conditional distribution.
   * The distribution must be normalized so that all the values for the given tail index sum to one.
   */
  template <typename Generator>
  std::vector<size_t> operator()(Generator& rng, const std::vector<size_t>& tail) const {
    assert(tail.size() <= psum_.arity());
    size_t nhead = psum_.arity() - tail.size();
    size_t nelem = psum_.shape().multiplier(nhead);
    const T* begin = psum_.begin() + psum_.shape().linear_back(tail);
    T p = std::uniform_real_distribution<T>()(rng);
    if (begin > psum_.begin()) { p += *(begin - 1); }
    size_t i = std::upper_bound(begin, begin + nelem, p) - begin;
    if (i < nelem) {
      return psum_.shape().index_front(i, nhead);
    } else {
      throw std::invalid_argument("The total probability is less than 1");
    }
  }

private:
  /// The Table of partial sums of probabilities.
  Table<T> psum_;

};

}
