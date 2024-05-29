#ifndef LIBGM_PROBABILITY_VECTOR_HPP
#define LIBGM_PROBABILITY_VECTOR_HPP

#include <libgm/math/exp.hpp>
#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

/**
 * A factor of a categorical probability distribution whose domain
 * consists of a single argument. The factor represents a non-negative
 * function directly with a parameter array \theta as f(X = x | \theta) =
 * \theta_x. In some cases, this class represents a array of probabilities
 * (e.g., when used as a prior in a hidden Markov model). In other cases,
 * e.g. in a pairwise Markov network, there are no constraints on the
 * normalization of f.
 *
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbabilityVector
  : Implements<
      Multiply<ProbabilityVector<T>, T>,
      Multiply<ProbabilityVector<T>, ProbabilityVector<T>>,
      MultiplyIn<ProbabilityVector<T>, T>,
      MultiplyIn<ProbabilityVector<T>, ProbabilityVector<T>>,
      Divide<ProbabilityVector<T>, T>,
      Divide<ProbabilityVector<T>, ProbabilityVector<T>>,
      DivideIn<ProbabilityVector<T>, T>,
      DivideIn<ProbabilityVector<T>, ProbabilityVector<T>>,
      Power<ProbabilityVector<T>>,
      Marginal<ProbabilityVector<T>>,
      Maximum<ProbabilityVector<T>>,
      Entropy<ProbabilityVector<T>, T>,
      KlDivergence<ProbabilityVector<T>, T>> {
public:
  // Public types
  //--------------------------------------------------------------------------
  using real_type = T;
  using result_type = T;
  using ll_type = ProbabilityVectorLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  ProbabilityVector() { }

  /// Constructs a factor with given arguments and uninitialized parameters.
  explicit ProbabilityVector(size_t length);

  /// Constructs a factor with the given arguments and constant value.
  ProbabilityVector(size_t length, T x);

  /// Constructs a factor with the given parameters.
  ProbabilityVector(const DenseVector<T>& param);

  /// Constructs a factor with the given parameters.
  ProbabilityVector(DenseVector<T>&& param);

  /// Constructs a factor with the given arguments and parameters.
  ProbabilityVector(std::initializer_list<T> params);

  /// Swaps the content of two ProbabilityVector factors.
  friend void swap(ProbabilityVector& f, ProbabilityVector& g) {
    std::swap(f.impl_, g.impl_);
  }

  /// Resets the content of this factor to the given arguments.
  void reset(size_t length);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of arguments of this factor.
  size_t arity() const {
    return 1;
  }

  /// Returns the total number of elements of the factor.
  size_t size() const;

  /**
   * Returns the pointer to the first parameter or nullptr if the factor is
   * empty.
   */
  T* begin();
  const RealType* begin() const;

  /**
   * Returns the pointer past the last parameter or nullptr if the factor is
   * empty.
   */
  T* end();
  const T* end() const

  /// Provides mutable access to the parameter array of this factor.
  DenseVector<T>& param();

  /// Returns the parameter array of this factor.
  const DenseVector<T>& param() const;

  /// Returns the value of the factor for the given row.
  Exp<T> operator()(size_t row) const {
    return Exp<T>(log(row));
  }

  /// Returns the value of the factor for the given assignment.
  Exp<T> operator()(const Assignment& a) const {
    return Exp<T>(log(a));
  }

  /// Returns the log-value of the factor for the given row.
  T log(size_t row) const;

  /// Returns the log-value of the factor for the given index.
  T log(const Assignment& a) const;

}; // class ProbabilityVector


} // namespace libgm

#endif
