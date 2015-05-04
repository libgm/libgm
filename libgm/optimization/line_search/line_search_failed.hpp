#ifndef LIBGM_LINE_SEARCH_FAILED_HPP
#define LIBGM_LINE_SEARCH_FAILED_HPP

#include <stdexcept>

namespace libgm {

  class line_search_failed : public std::runtime_error {
  public:
    line_search_failed(const std::string& reason)
      : std::runtime_error(reason) { }
  };

} // namespace libgm

#endif
