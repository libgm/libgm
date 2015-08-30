#ifndef LIBGM_RANGE_INTEGRAL_HPP
#define LIBGM_RANGE_INTEGRAL_HPP

#include <algorithm>
#include <numeric>
#include <vector>

namespace libgm {

  std::vector<std::size_t> range(std::size_t start, std::size_t stop) {
    std::vector<std::size_t> result;
    if (start < stop) {
      result.reserve(stop - start);
      while (start < stop) { result.push_back(start++); }
    }
    return result;
  }

  bool is_contiguous(const std::vector<std::size_t>& range) {
    auto pred = [] (std::size_t i, std::size_t j) { return i + 1 != j; };
    return std::adjacent_find(range.begin(), range.end(), pred) == range.end();
  }

} // namespace libgm

#endif
