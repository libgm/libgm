#ifndef LIBGM_NUMERIC_HPP
#define LIBGM_NUMERIC_HPP

#include <libgm/functional/arithmetic.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>

namespace libgm {

  /**
   * Returns the log-sum-exp of the given source range and stores the normalized
   * exponentiated values to the destination range starting at d_begin.
   */
  template <typename InForwardIt, typename OutForwardIt>
  typename std::iterator_traits<InForwardIt>::value_type
  log_sum_exp(InForwardIt s_begin, InForwardIt s_end, OutForwardIt d_begin) {
    using real_type = typename std::iterator_traits<InForwardIt>::value_type;

    // the sum of an empty range is 0, i.e., log-sum is -infinity
    if (s_begin == s_end) {
      return -std::numeric_limits<real_type>::infinity();
    }

    // compute the unnormalized probabilities and their sum
    real_type sum(0);
    real_type offset = *std::max_element(s_begin, s_end);
    OutForwardIt d_end = d_begin;
    for (InForwardIt it = s_begin; it != s_end; ++it, ++d_end) {
      *d_end = std::exp(*it - offset);
      sum += *d_end;
    }

    // normalize the probabilities and return the log-sum-exp
    std::transform(d_begin, d_end, d_begin, divided_by<real_type>(sum));
    return std::log(sum) + offset;
  }

  /**
   * Returns the log-sum-exp of the given source range.
   */
  template <typename InForwardIt>
  typename std::iterator_traits<InForwardIt>::value_type
  log_sum_exp(InForwardIt s_begin, InForwardIt s_end) {
    using real_type = typename std::iterator_traits<InForwardIt>::value_type;

    // the sum of an empty range is 0, i.e., log-sum is -infinity
    if (s_begin == s_end) {
      return -std::numeric_limits<real_type>::infinity();
    }

    // compute the sum of the unnormalized probabilities
    real_type sum(0);
    real_type offset = *std::max_element(s_begin, s_end);
    for (InForwardIt it = s_begin; it != s_end; ++it) {
      sum += std::exp(*it - offset);
    }

    // return the log-sum-exp
    return std::log(sum) + offset;
  }

} // namespace libgm

#endif
