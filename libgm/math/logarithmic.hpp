#ifndef LIBGM_LOGARITHMIC_HPP
#define LIBGM_LOGARITHMIC_HPP

#include <libgm/math/log_tag.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/serialization/serialize.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>

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
   * This is to prevent unintended conversions between T and logarithmic<T>.
   *
   * \todo specialize std::numeric_limits
   *
   * \ingroup math_number
   */
  template <typename T>
  struct logarithmic {
    static_assert(std::numeric_limits<T>::has_infinity,
                  "logarithmic<T> is only defined for T which have infinity");

    //! The log space representation of \f$x\f$, i.e., the value \f$\log x\f$.
    T lv;

    // Constructors
    //======================================================================

    /**
     * Default constructor. Initializes this object to exp(0) = 1.
     */
    logarithmic()
      : lv(0) { }

    /**
     * Log-space constructor.
     *
     * \param lv the logarithm of the value this object should represent
     * \param log_tag an indicator such as log_tag() that differentiates
     *        this constructor from the real-valued constructor below.
     *
     */
    logarithmic(T lv, log_tag)
      : lv(lv) { }

    /**
     * Constructor. The argumnet is the value to be represented, not its
     * logarithm.
     *
     * \param value the value this object should represent
     */
    explicit logarithmic(T value)
      : lv(std::log(value)) { }

    /**
     * Conversion constructor from a different logarithmic type.
     */
    template <typename U>
    explicit logarithmic(const logarithmic<U>& log)
      : lv(log.lv) { }

    /**
     * Conversion out of log space. Casting a log-space value into its
     * associated storage type computes the standard representation
     * from the log-space representation. This conversion is explicit,
     * meaning it must be invoked as in T(logarithmic_value).
     *
     * \return the value \f$x\f$, where this object represents
     *         \f$x\f$ in log-space
     */
    explicit operator T() const {
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
     * Returns the log space value representing the sum of this
     * value and the supplied log space value.
     *
     * This routine exploits a special purpose algorithm called log1p
     * that is in the C standard library. log1p(x) computes the value
     * \f$\log(1 + x)\f$ in a numerically stable way when \f$x\f$ is
     * close to zero.  Note that
     * \[
     *  \log(1 + y/x) + \log(x) = \log(x + y)
     * \]
     * Further note that
     * \[
     *  y/x = \exp(\log y - \log x)
     * \]
     * Thus, we first compute \f$y/x\f$ stably by choosing \f$x >
     * y\f$, and then use log1p to implement the first equation.
     *
     * \param a the value \f$\log x\f$
     * \return  the value \f$\log (x + y)\f$, where this object
     *          represents \f$\log y\f$
     */
    logarithmic operator+(const logarithmic& a) const {
      if (a.lv == -std::numeric_limits<T>::infinity()) {
        return *this;
      }
      if (lv == -std::numeric_limits<T>::infinity()) {
        return a;
      }
      T lx, ly;
      std::tie(ly, lx) = std::minmax(lv, a.lv);
      return logarithmic(std::log1p(std::exp(ly - lx)) + lx, log_tag());
    }

    /**
     * Returns the log space value representing the difference of this
     * value and the supplied log space value. This works by converting
     * into real-space and taking log on the result.
     */
    logarithmic operator-(const logarithmic& a) const {
      if (lv <= a.lv) {
        throw numerical_error(
          "logarithmic subtraction yields a negative value"
        );
      } else {
        return logarithmic(std::exp(lv) - std::exp(a.lv));
      }
    }

    /**
     * Returns the value representing the product of this value and
     * the supplied value.
     *
     * \param a the value \f$y\f$ represented in log-space
     * \return  the value \f$x \times y\f$ represented in log-space,
     *          where this object represents \f$x\f$ in log-space
     */
    logarithmic operator*(const logarithmic& a) const {
      return logarithmic(lv + a.lv, log_tag());
    }

    /**
     * Returns the value representing the ratio of this value and the
     * supplied value.
     *
     * \param a the value \f$y\f$ represented in log-space
     * \return  the value \f$x / y\f$, where this object
     *          represents \f$x\f$ in log-space
     */
    logarithmic operator/(const logarithmic& a) const {
      return logarithmic(lv - a.lv, log_tag());
    }

    /**
     * Updates this object to represent the sum of this value and the
     * supplied value.
     *
     * \param a the value \f$y\f$
     * \return  this value, after it has been updated to represent
     *          \f$x + y\f$ in log-space, where this object originally
     *          represented \f$x\f$ in log-space
     */
    logarithmic& operator+=(const logarithmic& y) {
      *this = *this + y;
      return *this;
    }

    /**
     * Returns the log space value representing the difference of this
     * value and the supplied log space value. This works by converting
     * into real-space and taking log on the result.
     */
    logarithmic& operator-=(const logarithmic& y) {
      *this = *this - y;
      return *this;
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
    logarithmic& operator*=(const logarithmic& y) {
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
    logarithmic& operator/=(const logarithmic& y) {
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
    bool operator==(const logarithmic& a) const {
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
    bool operator!=(const logarithmic& a) const {
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
    bool operator<(const logarithmic& a) const {
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
    bool operator>(const logarithmic& a) const {
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
    bool operator<=(const logarithmic& a) const {
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
    bool operator>=(const logarithmic& a) const {
      return lv >= a.lv;
    }

    /**
     * Returns the power of the logarithmic object raised to an exponent.
     */
    friend logarithmic pow(const logarithmic<T>& x, T exponent) {
      return logarithmic(x.lv * exponent, log_tag());
    }

    /**
     * Returns the logarithmic of the logaritmic object, i.e. lv.
     */
    friend T log(const logarithmic<T>& x) {
      return x.lv;
    }

    /**
     * Writes this log space representation to the supplied stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const logarithmic& x) {
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
    friend std::istream& operator>>(std::istream& in, logarithmic& x) {
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

  }; // struct logarithmic

  /**
   * Logarithmic value with double storage.
   */
  typedef logarithmic<double> logd;

} // namespace libgm

#endif

