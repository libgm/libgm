#ifndef LIBGM_NUMERIC_HPP
#define LIBGM_NUMERIC_HPP

#include <libgm/functional/arithmetic.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

namespace libgm {

  /**
   * Returns the log-sum-exp of the given source range and stores the normalized
   * exponentiated values to the destination range starting at d_begin.
   */
  template <typename InForwardIt, typename OutForwardIt>
  typename std::iterator_traits<InForwardIt>::value_type
  log_sum_exp(InForwardIt s_begin, InForwardIt s_end, OutForwardIt d_begin) {
    typedef typename std::iterator_traits<InForwardIt>::value_type real_type;
    if (s_begin == s_end) { return real_type(0); }

    // compute the offset used to avoid underflow
    real_type offset = *std::max_element(s_begin, s_end);

    // compute the unnormalized probabilities and their sum
    OutForwardIt d_end = d_begin;
    real_type sum(0);
    for (InForwardIt it = s_begin; it != s_end; ++it, ++d_end) {
      *d_end = std::exp(*it - offset);
      sum += *d_end;
    }

    // normalize the probabilities and return the log-sum-exp
    std::transform(d_begin, d_end, d_begin, divided_by<real_type>(sum));
    return std::log(sum) + offset;
  }

} // namespace libgm

#endif
