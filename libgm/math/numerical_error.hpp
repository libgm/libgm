#ifndef LIBGM_NUMERICAL_ERROR
#define LIBGM_NUMERICAL_ERROR

#include <stdexcept>

namespace libgm {

  struct numerical_error : public std::runtime_error {
    explicit numerical_error(const std::string& msg)
      : runtime_error(msg) { }
    explicit numerical_error(const char* msg)
      : runtime_error(msg) { }
  };

} // namespace libgm

#endif
