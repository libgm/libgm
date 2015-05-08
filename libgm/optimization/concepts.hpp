#ifndef LIBGM_OPTIMIZATION_CONCEPTS_HPP
#define LIBGM_OPTIMIZATION_CONCEPTS_HPP

#include <ostream>

namespace libgm {

  /**
   * Concept for a data structure for the variables being optimized over;
   * the data structure must essentially behave like a vector.
   * For many apps, the variables might be stored more naturally in a
   * collection of matrices and vectors, but this abstraction allows
   * optimization algorithms to treat the variables as a vector object.
   *
   * \ingroup optimization_concepts
   */
  template <typename Vec>
  struct OptimizationVector
    : DefaultConstructible<Vec>, CopyConstructible<Vec>, Assignable<Vec> {

    //! The type of values stored in this vector.
    typedef typename Vec::value_type value_type;

    //! Adds another vector to this one
    Vec& operator+=(const Vec& other);

    //! Subtracts another vector from this one
    Vec& operator-=(const Vec& other);

    //! Multiplication by a scalar value.
    Vec& operator*=(value_type d);

    //! Division by a scalar value.
    Vec& operator/=(value_type d);

    //! Adds a scalar multiple of a vector to another vector
    friend void axpy(value_type a, const Vec& x, Vec& y);

    //! Returns a vector whose each element is equal to the sign of the corresponding
    //! element in the input vector (-1 for negative, 0 for 0, 1 for positive).
    friend Vec sign(const Vec& x);

    //! Returns the inner product of two vectors
    friend value_type dot(const Vec& a, const Vec& b);

    //! Prints out the vector to an output stream
    friend std::ostream& operator<<(std::ostream& out, const Vec& x);

    concept_usage(OptimizationVector) {
      libgm::same_type(v += vcref, vref);
      libgm::same_type(v -= vcref, vref);
      libgm::same_type(v *= val, vref);
      libgm::same_type(v /= val, vref);
      libgm::same_type(dot(vcref, vcref), val);
      libgm::same_type(sign(vcref), v);
      libgm::same_type(out, out << vcref);
    }

  private:
    Vec v;
    static Vec& vref;
    static const Vec& vcref;
    value_type val;
    static std::ostream& out;

  }; // struct OptimizationVector

} // namespace libgm

#endif
