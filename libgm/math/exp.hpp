#pragma once

#include <libgm/serialization/serialize.hpp>

#include <cmath>
#include <iostream>
#include <limits>

namespace libgm {

/**
 * A numeric representation which represents the real number \f$x\f$
 * using a floating point representation of \f$\log x\f$.
 * This class allows us to represent values that are either too large
 * or too small (i.e., close to 0) in the regular representation,
 * or values that are by default represented in teh log scale
 * (such as the likelihood in an exponential family model).
 *
 * This class supports basic arithmetic operations. The conversion
 * to/from the real representation must be performed explicitly.
 * This is to prevent unintended conversions between T and Exp<T>.
 *
 * \todo specialize std::numeric_limits
 *
 * \ingroup math_number
 */
template <typename T>
struct Exp {
  static_assert(std::numeric_limits<T>::has_infinity,
                "Exp<T> is only defined for T which have infinity");

  /// The log space representation of \f$x\f$, i.e., the value \f$\log x\f$.
  T lv;

  // Constructors
  //======================================================================

  /**
   * Default constructor. Initializes this object to exp(0) = 1.
   */
  Exp()
    : lv(0) { }

  /**
   * Log-space constructor.
   *
   * \param lv the logarithm of the value this object should represent
   */
  explicit Exp(T lv)
    : lv(lv) { }

  /**
   * Conversion constructor from a different logarithmic type.
   */
  template <typename U>
  explicit Exp(const Exp<U>& log)
    : lv(log.lv) { }

  /**
   * Conversion out of log space. Casting a log-space value into its
   * associated storage type computes the standard representation
   * from the log-space representation.
   *
   * \return the value \f$x\f$, where this object represents \f$x\f$ in log-space
   */
  operator T() const {
    return std::exp(lv);
  }

  /**
   * Serializes the logarithmic to an archive.
   */
  void save(oarchive& ar) const {
    ar << lv;
  }

  /**
   * Deserialize the logarithmic from an archive.
   */
  void load(iarchive& ar) {
    ar >> lv;
  }

  // Arithmetic operations
  //======================================================================

  /**
   * Returns the value representing the product of this value and
   * the supplied value.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  the value \f$x \times y\f$ represented in log-space,
   *          where this object represents \f$x\f$ in log-space
   */
  Exp operator*(const Exp& a) const {
    return Exp(lv + a.lv, log_tag());
  }

  /**
   * Returns the value representing the ratio of this value and the
   * supplied value.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  the value \f$x / y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  Exp operator/(const Exp& a) const {
    return Exp(lv - a.lv, log_tag());
  }

  /**
   * Updates this object to represent the product of this value and
   * the supplied value.
   *
   * \param a the value \f$y\f$
   * \return  this value, after it has been updated to represent
   *          \f$x \times y\f$ in log-space, where this object
   *          originally represented \f$x\f$ in log-space
   */
  Exp& operator*=(const Exp& y) {
    lv += y.lv;
    return *this;
  }

  /**
   * Updates this object to represent the ratio of this value and
   * the supplied value.
   *
   * \param a the value \f$y\f$
   * \return  this value, after it has been updated to represent
   *          \f$x \times y\f$ in log-space, where this object
   *          originally represented \f$x\f$ in log-space
   */
  Exp& operator/=(const Exp& y) {
    lv -= y.lv;
    return *this;
  }

  /**
   * Returns true if this object represents the same value as the
   * supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x = y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator==(const Exp& a) const {
    return lv == a.lv;
  }

  /**
   * Returns true if this object represents a different value from
   * the supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x \neq y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator!=(const Exp& a) const {
    return lv != a.lv;
  }

  /**
   * Returns true if this object represents a smaller value than
   * the supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x < y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator<(const Exp& a) const {
    return lv < a.lv;
  }

  /**
   * Returns true if this object represents a larger value than
   * the supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x > y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator>(const Exp& a) const {
    return lv > a.lv;
  }

  /**
   * Returns true if this object represents a value that is less
   * than or equal to the supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x \le y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator<=(const Exp& a) const {
    return lv <= a.lv;
  }

  /**
   * Returns true if this object represents a value that is greater
   * than or equal to the supplied object.
   *
   * \param a the value \f$y\f$ represented in log-space
   * \return  true if \f$x \ge y\f$, where this object
   *          represents \f$x\f$ in log-space
   */
  bool operator>=(const Exp& a) const {
    return lv >= a.lv;
  }

  /**
   * Returns the power of the logarithmic object raised to an exponent.
   */
  friend Exp pow(const Exp& x, T exponent) {
    return Exp(x.lv * exponent);
  }

  /**
   * Returns the logarithmic of the logaritmic object, i.e. lv.
   */
  friend T log(const Exp& x) {
    return x.lv;
  }

  /**
   * Writes this log space representation to the supplied stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const Exp& x) {
    out << "exp(" << x.lv << ")";
    return out;
  }

  /**
   * Reads this log space value from the supplied stream.  There are
   * two accepted formats.  The first the same format used by the
   * value_type type; numbers in this format are converted into log
   * space representation.  The second format is 'exp(X)', where X
   * is in a format used by the value_type type.  In this case, the
   * read value is treated as a log space value.  For example,
   * reading '1.23e4' causes this object to represent the value
   * 1.23e4 in log space, as log(1.23e4); reading the value
   * exp(-1234.5) causes this object to represent the value
   * \f$e^{-1234.5}\f$, by storing the value 1234.5.
   *
   * @param in the stream from which this value is read
   */
  friend std::istream& operator>>(std::istream& in, Exp& x) {
    // Read off any leading whitespace.
    in >> std::ws;
    // Check to see if this value is written in log space.
    typedef typename std::istream::int_type int_t;
    if (in.peek() == static_cast<int_t>('e')) {
      in.ignore(4);
      in >> x.lv;
      in.ignore(1);
    } else {
      T val;
      in >> val;
      x.lv = std::log(val);
    }
    return in;
  }

}; // struct Exp

/**
 * Exp value with double storage.
 * \relates Exp
 */
using expd = Exp<double>;

/**
 * Exp value with float storage.
 * \relates Exp
 */
using expf = Exp<float>;

} // namespace libgm
