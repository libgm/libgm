#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/argument/values.hpp>
#include <Libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/exp.hpp>

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
  : Implements<
      // Direct operations
      Multiply<CanonicalGaussian, Exp<T>>,
      Multiply<CanonicalGaussian, CanonicalGaussian>,
      MultiplyIn<CanonicalGaussian, Exp<T>>,
      MultiplyIn<CanonicalGaussian, CanonicalGaussian>,
      Divide<CanonicalGaussian, Exp<T>>,
      Divide<CanonicalGaussian, CanonicalGaussian>,
      DivideIn<CanonicalGaussian, Exp<T>>,
      DivideIn<CanonicalGaussian, CanonicalGaussian>,

      // Join operations
      MultiplySpan<CanonicalGaussian>,
      MultiplySpanIn<CanonicalGaussian>,
      MultiplyDims<CanonicalGaussian>,
      MultiplyDimsIn<CanonicalGaussian>,
      DivideSpan<CanonicalGaussian>,
      DivideSpanIn<CanonicalGaussian>,
      DivideDims<CanonicalGaussian>,
      DivideDimsIn<CanonicalGaussian>,

      // Arithmetic operations
      Power<CanonicalGaussian>,

      // Aggregates
      Marginal<CanonicalGaussian, Exp<T>>,
      Maximum<CanonicalGaussian, Exp<T>>,
      MarginalSpan<CanonicalGaussian>,
      MarginalDims<CanonicalGaussian>,
      MaximumSpan<CanonicalGaussian>,
      MaximumDims<CanonicalGaussian>,

      // Normalization
      Normalize<CanonicalGaussian>,
      NormalizeHead<CanonicalGaussian>,

      // Restriction
      RestrictSpan<CanonicalGaussian>,
      RestrictDims<CanonicalGaussian>,

      // Entropy and divergences
      Entropy<CanonicalGaussian, T>,
      KlDivergence<CanonicalGaussian, T>
      MaxDifference<CanonicalGaussian, T>
    > {

public:
  static VTable vtable;

  // Factor member types
  using result_type = Exp<T>;

  /// Constructs an empty factor.
  CanonicalGaussian() = default;

  /// Constructs a factor with given implementation.
  CanonicalGaussian(ImplPtr ptr) : Implements(&vtable, std::move(ptr)) {}

  /// Constructs a canonical Gaussian factor equivalent to a constant.
  explicit CanonicalGaussian(Exp<T> value);

  /// Constructs a factor with given shape and constant value.
  CanonicalGaussian(Shape shape, Exp<T> value);

  /// Constructs a factor with the given shape and information vector / matrix.
  CanonicalGaussian(Shape shape, VectorType eta, MatrixType lambda, T lv = 0);

  /// Exchanges the content of two factors.
  friend void swap(CanonicalGaussian& f, CanonicalGaussian& g) {
    swap(f.impl_, g.impl_);
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of the factor.
  unsigned arity() const;

  /// Returns the log multiplier.
  T log_multiplier() const;

  /// Returns the information vector.
  const VectorType& inf_vector() const;

  /// Returns the information matrix.
  const MatrixType& inf_matrix() const;

  /// Evaluates the factor for the given vector.
  Exp<T> operator()(const Values& values) const;

  /// Returns the log-value of the factor for the given vector.
  T log(const Valuess& values) const;

  // Conversions
  //--------------------------------------------------------------------------

  MomentGausisan<T> moment() const;

}; // class CanonicalGaussian

} // namespace libgm
