#ifndef LIBGM_LOGARITHMIC_VECTOR_HPP
#define LIBGM_LOGARITHMIC_VECTOR_HPP

#include <libgm/math/exp.hpp>
#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include <iostream>
#include <numeric>

namespace libgm {

/**
  * A factor of a categorical logarithmic distribution whose domain
  * consists of a single argument. The factor represents a non-negative
  * function using the parameters \theta in the log space as f(X = x | \theta)=
  * exp(\theta_x). In some cases, this class represents a probability
  * distribution (e.g., when used as a prior in a hidden Markov model).
  * In other cases, e.g. in a pairwise Markov network, there are no constraints
  * on the normalization of f.
  *
  * \tparam T the type of values stored in the factor
  *
  * \ingroup factor_types
  * \see Factor
  */
template <typename T>
class LogarithmicVector
  : Implements<
      Multiply<LogarithmicVector<T>, Exp<T>>,
      Multiply<LogarithmicVector<T>, LogarithmicVector<T>>,
      MultiplyIn<LogarithmicVector<T>, Exp<T>>,
      MultiplyIn<LogarithmicVector<T>, LogarithmicVector<T>>,
      Divide<LogarithmicVector<T>, Exp<T>>,
      Divide<LogarithmicVector<T>, LogarithmicVector<T>>,
      DivideIn<LogarithmicVector<T>, Exp<T>>,
      DivideIn<LogarithmicVector<T>, LogarithmicVector<T>>,
      Power<LogarithmicVector<T>>,
      Marginal<LogarithmicVector<T>>,
      Maximum<LogarithmicVector<T>>,
      Entropy<LogarithmicVector<T>, T>,
      KlDivergence<LogarithmicVector<T>, T>> {
public:
  // Public types
  //--------------------------------------------------------------------------
  using real_type = T;
  using result_type = logarithmic<T>;
  using ll_type = LogarithmicVectorLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------
public:
  /// Default constructor. Creates an empty factor.
  LogarithmicVector() { }

  /// Constructs a factor with given arguments and uninitialized parameters.
  explicit LogarithmicVector(size_t length);

  /// Constructs a factor with the given arguments and constant value.
  LogarithmicVector(size_t length, Exp<T> x);

  /// Constructs a factor with the given parameters.
  LogarithmicVector(const DenseVector<T>& param);

  /// Constructs a factor with the given parameters.
  LogarithmicVector(DenseVector<T>&& param);

  /// Constructs a factor with the given arguments and parameters.
  LogarithmicVector(std::initializer_list<T> params);

  /// Swaps the content of two LogarithmicVector factors.
  friend void swap(LogarithmicVector& f, LogarithmicVector& g) {
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

}; // class LogarithmicVector


} // namespace libgm

#endif
