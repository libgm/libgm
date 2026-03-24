#pragma once

#include <libgm/datastructure/table.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

namespace libgm {

/**
 * A generator that returns a table filled with independent draws from
 * a random number distribution.
 *
 * \tparam Distribution
 *         A class that models the RandomNumberDistribution class.
 */
template <typename Distribution>
class TableGenerator {
public:
  // ParameterGenerator types
  using real_type = typename Distribution::result_type;
  using result_type = Table<real_type>;
  using shape_type = Shape;
  using param_type = typename Distribution::param_type;

  /**
   * Constructs a TableGenerator, passing the parameters down to
   * the distribution.
   */
  template <typename... Arg>
  TableGenerator(Arg&&... arg)
    : distribution_(std::forward<Arg>(arg)...) { }

  /**
   * Returns the parameter set associated with the distribution.
   */
  param_type param() const {
    return distribution_.param();
  }

  /**
   * Sets the parameter set associated with the distribution.
   */
  void param(const param_type& params) {
    distribution_.param(params);
  }

  /**
   * Prints the generator to an output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const TableGenerator& g) {
    out << "TableGenerator(" << g.distribution_ << ")";
    return out;
  }

  /**
   * Generates a table using the stored random number distribution.
   */
  template <typename Generator>
  Table<real_type> operator()(Shape shape, Generator& g) {
    Table<real_type> r(std::move(shape));
    std::generate(r.begin(), r.end(), std::bind(distribution_, std::ref(g)));
    return r;
  }

private:
  Distribution distribution_;
};

/**
 * A TableGenerator that draws the table elements from a uniform
 * distribution.
 *
 * \relates TableGenerator
 */
template <typename RealType = double>
using UniformTableGenerator = TableGenerator<std::uniform_real_distribution<RealType>>;

/**
 * A TableGenerator that returns a table whose parameters are drawn from a gamma distribution.
 *
 * In the special case, when the second parameter of the gamma distribution is
 * 1, and the table is normalized to sum to 1, resulting in a draw from
 * a Dirichlet distribution.
 *
 * \relates TableGenerator
 */
template <typename RealType = double>
using DirichletTableGenerator = TableGenerator<std::gamma_distribution<RealType>>;

/**
 * A generator that draws a random table that is a sum of a fixed base number and a constant diagonal matrix, whose
 * value is drawn from a uniform distribution. This can be used to generate associative factors and ising factors.
 */
template <typename Distribution>
class DiagonalTableGenerator {
public:
  using real_type = typename Distribution::result_type;
  using result_type = Table<real_type>;
  using shape_type = Shape;

  /// Constructs a diagonal_TableGenerator, passing the arguments down to the diagonal_generator_param struct.
  template <typename... Arg>
  DiagonalTableGenerator(real_type base, Arg&&... arg)
    : base_(base), distribution_(std::forward<Arg>(arg)...) { }

  /// Prints the generator to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const DiagonalTableGenerator& g) {
    out << "DiagonalTableGenerator(" << g.base_ << ", " << g.distribution_ << ")";
    return out;
  }

  /// Generates a table using the stored random number distribution.
  template <typename Generator>
  Table<real_type> operator()(unsigned arity, size_t n, Generator& g) {
    Table<real_type> result(Shape(arity, n), base_);
    std::vector<size_t> index;
    for (size_t k = 0; k < n; ++k) {
      index.assign(arity, k);
      result(index) += distribution_(g);
    }
    return result;
  }

private:
  real_type base_;
  Distribution distribution_;
};

}
