#ifndef LIBGM_VOID_HPP
#define LIBGM_VOID_HPP

#include <cstddef>
#include <iosfwd>

namespace libgm {

  // Forward declarations
  class iarchive;
  class oarchive;

  // empty type (useful primarily for empty vertex and edge properties)
  struct void_ { };

  /**
   * Equality comparison for void_.
   * \relates void_
   */
  inline bool operator==(const void_&, const void_&) {
    return true;
  }

  /**
   * Inequality comparison for void_.
   * \relates void_
   */
  inline bool operator!=(const void_&, const void_&) {
    return false;
  }

  /**
   * Less than comparison for void_.
   * \relates void_
   */
  inline bool operator<(const void_&, const void_&) {
    return false;
  }

  /**
   * Prints a void to an output stream.
   * \relates void_
   */
  inline std::ostream& operator<<(std::ostream& out, void_) {
    return out;
  }

  /**
   * Serializes a void to an output archive.
   * \relates void_
   */
  inline oarchive& operator<<(oarchive& ar, const void_&) {
    return ar;
  }

  /**
   * Deserializes a void from an output archive.
   * \relates void_
   */
  inline iarchive& operator>>(iarchive& ar, void_&) {
    return ar;
  }

} // namespace libgm

#endif
