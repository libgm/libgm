#ifndef LIBGM_PROBABILITY_TABLE_HPP
#define LIBGM_PROBABILITY_TABLE_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/exp.hpp>
#include <libgm/math/numeric.hpp>
#include <libgm/math/likelihood/canonical_table_ll.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/math/tags.hpp>

#include <initializer_list>
#include <iostream>
#include <random>
#include <type_traits>

namespace libgm {

/**
 * A factor of a categorical probability distribution in the probability
 * space. This factor represents a non-negative function over finite
 * arguments X directly using its parameters, f(X = x | \theta) = \theta_x.
 * In some cases, e.g. in a Bayesian network, this factor in fact
 * represents a (conditional) probability distribution. In other cases,
 * e.g. in a Markov network, there are no constraints on the normalization
 * of f.
 *
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 * \see Factor
 */
template <typename T>
class ProbablityTable
  : Implements<
      Assign<ProbablityTable<T>, T>,
      Assign<ProbablityTable<T>, ProbablityTable<T>>,
      Multiply<ProbablityTable<T>, T>,
      Multiply<ProbablityTable<T>, ProbablityTable<T>>,
      MultiplyIn<ProbablityTable<T>, T>,
      MultiplyIn<ProbablityTable<T>, ProbablityTable<T>>,
      MultiplySpan<ProbablityTable<T>>,
      MultiplySpanIn<ProbablityTable<T>>,
      MultiplyList<ProbablityTable<T>>,
      MultiplyListIn<ProbablityTable<T>>,
      Divide<ProbablityTable<T>, T>,
      Divide<ProbablityTable<T>, ProbablityTable<T>>,
      DivideIn<ProbablityTable<T>, T>,
      DivideIn<ProbablityTable<T>, ProbablityTable<T>>,
      DivideSpan<ProbablityTable<T>>,
      DivideSpanIn<ProbablityTable<T>>,
      DivideList<ProbablityTable<T>>,
      DivideListIn<ProbablityTable<T>>,
      Power<ProbablityTable<T>>,
      Marginal<ProbablityTable<T>>,
      Maximum<ProbablityTable<T>>,
      Entropy<ProbablityTable<T>, T>,
      KlDivergence<ProbablityTable<T>, T>> {

public:
  // Public types
  //--------------------------------------------------------------------------

  using ll_type = CanonicalTableLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  ProbablityTable() = default;

  /// Constructs a factor equivalent to a constant.
  explicit ProbablityTable(T value);

  /// Constructs a factor with the given shape and constant value.
  explicit ProbablityTable(const std::vector<size_t>& shape, T value = T(1));

  /// Creates a factor with the specified shape and parameters.
  ProbablityTable(const std::vector<size_t>& shape, std::initializer_list<t> values);

  /// Creates a factor with the specified parameters.
  ProbablityTable(const Table<T>& param);

  /// Creates a factor with the specified parameters.
  ProbablityTable(Table<T>&& param);

  /// Exchanges the content of two factors.
  friend void swap(ProbablityTable& f, ProbablityTable& g) {
    std::swap(f.impl_, g.impl_);
  }

  /**
   * Resets the content of this factor to the given sequence of arguments.
   * If the table size changes, the table elements become invalidated.
   */
  void reset(const std::vector<size_t>& shape);

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of dimensions (guaranteed to be constant-time).
  size_t arity() const;

  /// Returns the total number of elements of the table.
  size_t size() const;

  /// Returns the shape of the underlying table.
  const std::vector<size_t>& shape() const;

  /**
   * Returns the pointer to the first parameter or nullptr if the factor
   * is empty.
   */
  RealType* begin();
  const RealType* begin() const;

  /**
   * Returns the pointer past the last parameter or nullptr if the factor
   * is empty.
   */
  RealType* end();
  const RealType* end() const;

  /// Provides mutable access to the parameter table of this factor.
  Table<T>& param();

  /// Returns the parameter table of this factor.
  const Table<T>& param() const;

  /// Returns the value of the expression for the given index.
  Exp<T> operator()(const Assignment& a) const;

  /// Returns the log-value of the expression for the given index.
  T log(const Assignment& a) const;

}; // class ProbablityTable

} // namespace libgm

#endif
