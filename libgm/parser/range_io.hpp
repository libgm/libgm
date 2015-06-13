#ifndef LIBGM_RANGE_IO_HPP
#define LIBGM_RANGE_IO_HPP

#include <iostream>

namespace libgm {

  template <typename InputIterator>
  void print_range(std::ostream& out,
                   InputIterator begin, InputIterator end,
                   char left, char mid, char right) {
    out << left;
    bool first = true;
    while (begin != end) {
      if (first) { first = false; } else { out << mid; }
      out << *begin;
      ++begin;
    }
    out << right;
  }

} // namespace libgm

#endif

