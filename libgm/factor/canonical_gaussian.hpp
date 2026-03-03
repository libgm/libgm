#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/assignment/real_values.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/exp.hpp>
#include <libgm/math/eigen/dense.hpp>

namespace libgm {

// Forward declaration of the factor
template <typename T> class MomentGaussian;

/**
 * A factor of a multivariate normal (Gaussian) distribution in the natural
 * parameterization of the exponential family. Given an information vector
 * \eta and information matrix \lambda, this factor represents an
 * exponentiated quadratic function exp(-0.5 * x^T \lambda x + x^T \eta + a).
 *
 * \tparam T The real type representing the parameters.
 * \ingroup factor_types
 */
template <typename T>
class CanonicalGaussian
  : public Object,
    public Implements<
      // Direct operations
      Multiply<CanonicalGaussian<T>, Exp<T>>,
      Multiply<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplyIn<CanonicalGaussian<T>, Exp<T>>,
      MultiplyIn<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      Divide<CanonicalGaussian<T>, Exp<T>>,
      Divide<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideIn<CanonicalGaussian<T>, Exp<T>>,
      DivideIn<CanonicalGaussian<T>, CanonicalGaussian<T>>,

      // Join operations
      MultiplySpan<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplyDims<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplyInSpan<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplyInDims<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideSpan<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideDims<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideInSpan<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideInDims<CanonicalGaussian<T>, CanonicalGaussian<T>>,

      // Arithmetic operations
      Power<CanonicalGaussian<T>, T>,
      WeightedUpdate<CanonicalGaussian<T>, T>,

      // Aggregates
      Marginal<CanonicalGaussian<T>, Exp<T>>,
      Maximum<CanonicalGaussian<T>, Exp<T>, RealValues<T>>,
      MarginalSpan<CanonicalGaussian<T>>,
      MarginalDims<CanonicalGaussian<T>>,
      MaximumSpan<CanonicalGaussian<T>>,
      MaximumDims<CanonicalGaussian<T>>,

      // Normalization
      Normalize<CanonicalGaussian<T>>,
      NormalizeHead<CanonicalGaussian<T>>,

      // Restriction
      RestrictSpan<CanonicalGaussian<T>, RealValues<T>>,
      RestrictDims<CanonicalGaussian<T>, RealValues<T>>,

      // Entropy and divergences
      Entropy<CanonicalGaussian<T>, T>,
      KlDivergence<CanonicalGaussian<T>, T>,
      MaxDifference<CanonicalGaussian<T>, T>
    > {
public:
  // Factor member types
  using result_type = Exp<T>;

  // Implementation class.
  struct Impl;

  // Function table.
  static const typename CanonicalGaussian::VTable vtable;

  /// Constructs an empty factor.
  CanonicalGaussian() = default;

  /// Constructs a canonical Gaussian factor equivalent to a constant.
  explicit CanonicalGaussian(Exp<T> value);

  /// Constructs a factor with given shape and constant value.
  explicit CanonicalGaussian(Shape shape, Exp<T> value = Exp<T>(0));

  /// Constructs a factor with the given shape and information vector / matrix.
  CanonicalGaussian(Shape shape, Vector<T> eta, Matrix<T> lambda, T lv = 0);

  /// Exchanges the content of two factors.
  friend void swap(CanonicalGaussian& f, CanonicalGaussian& g) {
    swap(f.impl_, g.impl_);
  }

  /// Initializes this factor to the given shape.
  void reset(Shape shape);

  /// Initializes this factor to the given implementation object, to be owned by this.
  void reset(Impl* impl);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of the factor.
  unsigned arity() const;

  /// Returns the shape of the factor.
  const Shape& shape() const;

  /// Returns the log multiplier.
  T log_multiplier() const;

  /// Returns the information vector.
  const Vector<T>& inf_vector() const;

  /// Returns the information matrix.
  const Matrix<T>& inf_matrix() const;

  /// Evaluates the factor for the given vector.
  Exp<T> operator()(const RealValues<T>& values) const;

  /// Returns the log-value of the factor for the given vector.
  T log(const RealValues<T>& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  MomentGaussian<T> moment() const;

private:
  Impl& impl();
  const Impl& impl() const;

}; // class CanonicalGaussian

} // namespace libgm
