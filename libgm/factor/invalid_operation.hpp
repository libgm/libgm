#ifndef LIBGM_INVALID_OPERATION_HPP
#define LIBGM_INVALID_OPERATION_HPP

#include <stdexcept>

namespace libgm {

  /**
   * An exception thrown when an operation cannot be performed on a
   * factor.
   *
   * @see canonical_gaussian::collapse, moment_gaussian::restrict
   *
   * \ingroup factor_exceptions
   */
  class invalid_operation : public std::runtime_error {
  public:
    explicit invalid_operation(const std::string& what)
      : std::runtime_error(what) { }
  };

}

#endif
