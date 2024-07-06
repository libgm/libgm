#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/factor/implements.hpp>
#include <libgm/factor/interfaces.hpp>
#include <libgm/math/eigen/dense.hpp>
// #include <libgm/math/likelihood/softmax_ll.hpp>
// #include <libgm/math/likelihood/softmax_mle.hpp>

#include <iostream>

namespace libgm {

// Forward declaration
template <typename T> class ProbabilityVector;

/**
 * A softmax function over one discrete variable y and a vector of
 * real-valued features x. This function is equal to a normalized
 * exponential, p(y=i | x) = exp(b_i + w_i^T x) / sum_j exp(b_j + w_j^T x).
 * Here, b is a bias vector and w is a weight matrix with rows w_i^T.
 * The parameter matrices are dense, but the function can be evaluated
 * on sparse feature vectors.
 *
 * \tparam T a real type for representing each parameter.
 */
template <typename RealType = double>
class Softmax
  : public Implements<RestrictSpan<Softmax, ProbabilityVector<T>>> {
  // Public types
  //==========================================================================
public:
  // Factor member types
  using result_type = T;

  // LearnableFactor member types
  // typedef softmax_ll<T>  ll_type;
  // typedef softmax_mle<T> mle_type;

  // Constructors and conversion operators
  //==========================================================================

  /// Default constructor. Creates an empty factor.
  Softmax() = default;

  /// Constructs a factor with the given tail shape.
  Softmax(Shape tail_shape, DenseMatrix<T> weight, DenseVector<T> bias);

  /// Exchanges the two factors.
  friend void swap(Softmax& f, Softmax& g) {
    std::swap(f.impl_, g.impl_);
  }

  // Accessors and comparison operators
  //==========================================================================

  /// Returns the number of head arguments (always 1).
  size_t head_arity() const { return 1; }

  /// Returns the number of tail arguments.
  size_t tail_arity() const;

  /// Returns the number of arguments.
  size_t arity() const;

  /// Returns the number of labels.
  size_t num_labels() const;

  /// Returns the tail shape.
  const Shape& tail_shape() const;

  /// Returns the weight matrix.
  const DenseMatrix<T>& weight() const;

  /// Returns the bias vector.
  const DenseVector<T>& bias() const;

  /// Returns the value of the factor for the given index.
  Exp<T> operator()(size_t label, const Values& features) const;

  /// Returns the log-value of this factor for the given index.
  T log(size_t label, const Values& features) const;

}; // class Softmax

} // namespace libgm

#endif
