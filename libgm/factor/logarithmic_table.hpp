#ifndef LIBGM_LOGARITHMIC_TABLE_HPP
#define LIBGM_LOGARITHMIC_TABLE_HPP

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
 * A factor of a categorical distribution represented in the log space.
 * This factor represents a non-negative function over finite variables
 * X as f(X | \theta) = exp(\sum_x \theta_x * 1(X=x)). In some cases,
 * e.g. in a Bayesian network, this factor also represents a probability
 * distribution in the log-space.
 *
 * \tparam RealType a real type representing each parameter
 *
 * \ingroup factor_types
 */
template <typename T>
class LogarithmicTable
  : Implements<
      Assign<LogarithmicTable<T>, Exp<T>>,
      Assign<LogarithmicTable<T>, LogarithmicTable<T>>,
      Multiply<LogarithmicTable<T>, Exp<T>>,
      Multiply<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplyIn<LogarithmicTable<T>, Exp<T>>,
      MultiplyIn<LogarithmicTable<T>, LogarithmicTable<T>>,
      MultiplySpan<LogarithmicTable<T>>,
      MultiplySpanIn<LogarithmicTable<T>>,
      MultiplyList<LogarithmicTable<T>>,
      MultiplyListIn<LogarithmicTable<T>>,
      Divide<LogarithmicTable<T>, Exp<T>>,
      Divide<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideIn<LogarithmicTable<T>, Exp<T>>,
      DivideIn<LogarithmicTable<T>, LogarithmicTable<T>>,
      DivideSpan<LogarithmicTable<T>>,
      DivideSpanIn<LogarithmicTable<T>>,
      DivideList<LogarithmicTable<T>>,
      DivideListIn<LogarithmicTable<T>>,
      Power<LogarithmicTable<T>>,
      Marginal<LogarithmicTable<T>>,
      Maximum<LogarithmicTable<T>>,
      Entropy<LogarithmicTable<T>, T>,
      KlDivergence<LogarithmicTable<T>, T>> {

public:
  // Public types
  //--------------------------------------------------------------------------

  using ll_type = CanonicalTableLL<T>;

  // Constructors and conversion operators
  //--------------------------------------------------------------------------

  /// Default constructor. Creates an empty factor.
  LogarithmicTable() = default;

  /// Constructs a factor equivalent to a constant.
  explicit LogarithmicTable(Exp<T> value);

  /// Constructs a factor with the given shape and constant value.
  explicit LogarithmicTable(const std::vector<size_t>& shape, Exp<T> value = Exp<T>(0));

  /// Creates a factor with the specified shape and parameters.
  LogarithmicTable(const std::vector<size_t>& shape, std::initializer_list<t> values);

  /// Creates a factor with the specified parameters.
  LogarithmicTable(const Table<T>& param);

  /// Creates a factor with the specified parameters.
  LogarithmicTable(Table<T>&& param);

  /// Exchanges the content of two factors.
  friend void swap(LogarithmicTable& f, LogarithmicTable& g) {
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

}; // class LogarithmicTable

} // namespace libgm

#endif
