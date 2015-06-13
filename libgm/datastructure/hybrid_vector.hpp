#ifndef LIBGM_HYBRID_INDEX_HPP
#define LIBGM_HYBRID_INDEX_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/parser/range_io.hpp>

namespace libgm {

  /**
   * A class that represents a dense vector with an integral and
   * a real component.
   *
   * \tparam T the real type representing the vector elements
   */
  template <typename T = double>
  struct hybrid_vector
    : public uint_vector, real_vector<T> {

    //! Default constructor; constructs empty index.
    hybrid_vector() { }

    /**
     * Constructs an index with the integral and real components
     * of specified lengths. The integral component is initialized
     * to a 0, but the vector one is not.
     */
    hybrid_vector(std::size_t nint, std::size_t nreal)
      : uint_vector(nint),
        real_vector<T>(nreal) { }

    //! Constructs an index with the given finite and vector component.
    hybrid_vector(const uint_vector& uint,
                  const real_vector<T>& real)
      : uint_vector(uint),
        real_vector<T>(real) { }

    //! Returns the integral component..
    uint_vector& uint() {
      return *this;
    }

    //! Returns the integral component.
    const uint_vector& uint() const {
      return *this;
    }

    //! Returns the real component.
    real_vector<T>& real() {
      return *this;
    }

    //! Returns the real component.
    const real_vector<T>& real() const {
      return *this;
    }

    //! Returns the length of the integral component.
    std::size_t uint_size() const {
      return uint().size();
    }

    //! Returns the length of the real component.
    std::size_t real_size() const {
      return real().size();
    }

    //! Swaps the contents of two hybrid indices.
    friend void swap(hybrid_vector& a, hybrid_vector& b) {
      a.uint().swap(b.uint());
      a.real().swap(b.real());
    }

    //! Resizes the index to the given lengths. The original values may be lost.
    void resize(std::size_t nint, std::size_t nreal) {
      uint().resize(nint);
      real().resize(nreal);
    }

    //! Returns true if the two hybrid indices are equal
    friend bool operator==(const hybrid_vector& a, const hybrid_vector& b) {
      return a.uint() == b.uint() && a.real() == b.real();
    }

    //! Returns true if the two hybrid indices are not equal.
    friend bool operator!=(const hybrid_vector& a, const hybrid_vector& b) {
      return !(a == b);
    }

  }; // struct hybrid_vector

  /**
   * Prints a hybrid vector to an output stream.
   * \relates hybrid_vector
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const hybrid_vector<T>& v) {
    print_range(out, v.uint().begin(), v.uint().end(), '[', ' ', ']');
    out << ' ';
    print_range(out, v.real().data(), v.real().data() + v.real().size(),
                '[', ' ', ']');
    return out;
  }

} // namespace libgm

#endif
