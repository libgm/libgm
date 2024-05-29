#ifndef LIBGM_FACTOR_CANONICAL_GAUSSIAN_HPP
#define LIBGM_FACTOR_CANONICAL_GAUSSIAN_HPP

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
      Assign<CanonicalGaussian<T>, Exp<T>>,
      Assign<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      Multiply<CanonicalGaussian<T>, Exp<T>>,
      Multiply<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplyIn<CanonicalGaussian<T>, Exp<T>>,
      MultiplyIn<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      MultiplySpan<CanonicalGaussian<T>>,
      MultiplySpanIn<CanonicalGaussian<T>>,
      MultiplyList<CanonicalGaussian<T>>,
      MultiplyListIn<CanonicalGaussian<T>>,
      Divide<CanonicalGaussian<T>, Exp<T>>,
      Divide<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideIn<CanonicalGaussian<T>, Exp<T>>,
      DivideIn<CanonicalGaussian<T>, CanonicalGaussian<T>>,
      DivideSpan<CanonicalGaussian<T>>,
      DivideSpanIn<CanonicalGaussian<T>>,
      DivideList<CanonicalGaussian<T>>,
      DivideListIn<CanonicalGaussian<T>>,
      Power<CanonicalGaussian<T>>,
      Marginal<CanonicalGaussian<T>>,
      Maximum<CanonicalGaussian<T>>,
      Entropy<CanonicalGaussian<T>, T>,
      KlDivergence<CanonicalGaussian<T>, T>> {

public:
  // Factor member types
  using real_type = T;
  using result_type = Exp<T>;
  using ImplPtr = std::unique_ptr<Impl<CanonicalGaussian>>;

  /// Constructs an empty factor.
  CanonicalGaussian() = default;

  /// Move constructor.
  CanonicalGaussian(CanonicalGaussian&& other) = default;

  /// Copy constructor.
  CanonicalGaussian(const CanonicalGaussian& other);

  /// Constructs a factor with given implementation.
  CanonicalGaussian(ImplPtr ptr) : impl_(std::move(ptr)) {}

  /// Constructs a canonical Gaussian factor equivalent to a constant.
  explicit CanonicalGaussian(Exp<T> value);

  /// Constructs a canonical Gaussian factor from a moment Gaussian.
  explicit CanonicalGaussian(const MomentGaussian<T>& mg);

  /// Constructs a factor with given arity and constant value.
  CanonicalGaussian(unsigned arity, Exp<T> value);

  /// Constructs a factor with the given information vector and matrix.
  CanonicalGaussian(const VectorType& eta, const MatrixType& lambda, T lv = 0);

  /// Constructs a factor with the given information vector and matrix.
  CanonicalGaussian(VectorType&& eta, MatrixType&& lambda, T lv = 0);

  /// Destructor.
  ~CanonicalGaussian();

  /// Returns true if the factor has no data. This is different from arity() == 0.
  bool empty() const {
    return !impl_;
  }

  /// Exchanges the content of two factors.
  friend void swap(CanonicalGaussian& f, CanonicalGaussian& g) {
    swap(f.impl_, g.impl_);
  }

  /// Returns the log multiplier.
  RealType log_multiplier() const;

  /// Returns the information vector.
  const VectorType& inf_vector() const;

  /// Returns the information matrix.
  const MatrixType& inf_matrix() const;

private:
  //! The parameters of the factor, encapsulated as a struct.
  std::unique_ptr<Impl<CanonicalGaussian>> impl_;

}; // class canonical_gaussian

} // namespace libgm

#endif
