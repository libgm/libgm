#ifndef LIBGM_REAL_PAIR_HPP
#define LIBGM_REAL_PAIR_HPP

#include <iostream>
#include <tuple>
#include <utility>

namespace libgm {

  /**
   * A class that mimics std::pair for real types, adding support for
   * basic arithmetic operations that treat the pair as a vector with
   * two elements.
   *
   * \tparam T a type that supports real-valued arithmic operations.
   */
  template <typename T = double>
  struct real_pair {
    // Useful typedefs
    typedef T first_type;
    typedef T second_type;
    typedef T real_type;

    // Members
    T first;
    T second;

    // Standard operations
    //==========================================================================

    //! Default constructor. Value-initializes both elements.
    real_pair()
      : first(0), second(0) { }

    //! Initializes first with a and second with b.
    real_pair(T a, T b)
      : first(a), second(b) { }

    //! Converts an std::pair to a real_pair.
    real_pair(const std::pair<T,T>& p)
      : first(p.first), second(p.second) { }

    //! Converts a pair to a tuple of references, in order to support std::tie.
    operator std::tuple<T&, T&>() {
      return std::tuple<T&, T&>(first, second);
    }

    //! Swaps the contents of two pairs.
    friend void swap(real_pair& x, real_pair& y) {
      using std::swap;
      swap(x.first, y.first);
      swap(x.second, y.second);
    }

    //! Returns true if two real_pair objects are equal.
    friend bool operator==(const real_pair& x, const real_pair& y) {
      return x.first == y.first && x.second == y.second;
    }

    //! Returns true if two rea_pair objects are not equal.
    friend bool operator!=(const real_pair& x, const real_pair& y) {
      return x.first != y.first || x.second != y.second;
    }

    //! Prints the real_pair to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const real_pair& p) {
      out << '(' << p.first << ',' << p.second << ')';
      return out;
    }

    // Vector operations
    //==========================================================================

    //! Adds another pair to this one.
    real_pair& operator+=(const real_pair& x) {
      first += x.first;
      second += x.second;
      return *this;
    }

    //! Subtracts another pair from this one.
    real_pair& operator-=(const real_pair& x) {
      first -= x.first;
      second -= x.second;
      return *this;
    }

    //! Multiplies the pair by a constant.
    real_pair& operator*=(T a) {
      first *= a;
      second *= a;
      return *this;
    }

    //! Divides the pair by a constant.
    real_pair& operator/=(T a) {
      first /= a;
      second /= a;
      return *this;
    }

    //! Adds two pairs.
    friend real_pair operator+(const real_pair& x, const real_pair& y) {
      return real_pair(x.first + y.first, x.second + y.second);
    }

    //! Subtracts two pairs.
    friend real_pair operator-(const real_pair& x, const real_pair& y) {
      return real_pair(x.first - y.first, x.second - y.second);
    }

    //! Multiplies a pair by a constant.
    friend real_pair operator*(const real_pair& x, T a) {
      return real_pair(x.first * a, x.second * a);
    }

    //! Multiplies a pair by a constant.
    friend real_pair operator*(T a, const real_pair& x) {
      return real_pair(x.first * a, x.second * a);
    }

    //! Divides a pair by a constant.
    friend real_pair operator/(const real_pair& x, T a) {
      return real_pair(x.first / a, x.second / a);
    }

  }; // class real_pair

} // namespace libgm

#endif
