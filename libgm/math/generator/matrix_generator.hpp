#pragma once

#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/generator/diagonal_generator_param.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

namespace libgm {

/**
 * A generator that returns a matrix filled with independent draws from a random number distribution.
 *
 * \tparam Distribution
 *         A class that models the RandomNumberDistribution class.
 */
template <typename Distribution>
class MatrixGenerator {
  // ParameterGenerator types
  using real_type = typename Distribution::real_type;
  using result_type = Matrix<real_type>;
  using shape_type = std::pair<size_t, size_t>;
  using param_type = typename Distribution::param_type;

  /// Constructs a MatrixGenerator, passing the parameters down to the distribution.
  template <typename... Arg>
  MatrixGenerator(Arg&&... arg)
    : distribution_(std::forward<Arg>(arg)...) { }

  /// Returns the parameter set associated with the distribution.
  param_type param() const {
    return distribution_.param();
  }

  /// Sets the parameter set associated with the distribution.
  void param(const param_type& params) {
    distribution_.param(params);
  }

  /// Prints the generator to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const MatrixGenerator& g) {
    out << "MatrixGenerator(" << g.distribution_ << ")";
    return out;
  }

  /// Generates a matrix using the stored random number distribution.
  template <typename Generator>
  Matrix<real_type> operator()(size_t rows, size_t cols, Generator& g) const {
    Matrix<real_type> r(rows, cols);
    std::generate(r.data(), r.data() + r.size(), std::bind(distribution_, std::ref(g)));
    return r;
  }

  /// Generates a matrix using the stored random number distribution.
  template <typename Generator>
  Matrix<real_type> operator()(std::pair<size_t, size_t> shape, Generator& g) const {
    return operator()(shape.first, shape.second, g);
  }

private:
  Distribution distribution_;
};

/**
 * A MatrixGenerator that draws the matrix elements from a uniform
 * distribution.
 * \relates MatrixGenerator
 */
template <typename T = double>
using UniformMatrixGenerator = MatrixGenerator<std::uniform_distribution<T>>;

/**
 * A MatrixGenerator that returns a matrix whose parameters are drawn from a gamma distribution.
 *
 * In the special case, when the second parameter of the gamma distribution is
 * 1, and the matrix is normalized to sum to 1, this results in a draw from
 * a dirichlet distribution.
 *
 * \relates MatrixGenerator
 */
template <typename T = double>
using DirichlatMatrixGenerator = MatrixGenerator<std::gamma_distribution<T> >;

/**
 * A generator that draws a random matrix that is a sum of a fixed base
 * number and a constant diagonal matrix, whose value is drawn from
 * a uniform distribution. This can be used to generate associative
 * factors and ising factors.
 */
template <typename RealType = double>
class DiagonalMatrixGenerator {
public:
  // ParameterGenerator types
  using real_type = RealType;
  using result_type = Matrix<RealType>;
  using shape_type = size_t;
  using param_type = diagonal_generator_param<RealType>;

  /**
   * Constructs a DiagonalMatrixGenerator, passing the arguments down to
   * the diagonal_generator_param struct.
   */
  template <typename... Arg>
  DiagonalMatrixGenerator(Arg&&... arg)
    : param_(std::forward<Arg>(arg)...) { }

  /**
   * Returns the parameter set associated with the distribution.
   */
  param_type param() const {
    return param_;
  }

  /**
   * Sets the parameter set associated with the distribution.
   */
  void param(const param_type& params) {
    param_ = params;
  }

  /**
   * Prints the generator to an output stream.
   */
  friend std::ostream&
  operator<<(std::ostream& out, const MatrixGenerator& g) {
    out << "DiagonalMatrixGenerator(" << g.param_ << ")";
    return out;
  }

  /**
   * Generates a matrix using the stored random number distribution.
   */
  template <typename Generator>
  Matrix<real_type> operator()(size_t n, Generator& g) const {
    std::uniform_real_distribution<RealType> offset(param_.lower,
                                                    param_.upper);
    return Matrix<RealType>::Constant(n, n, param_.base) +
      Matrix<RealType>::Identity(n, n, offset(g));
  }

private:
  diagonal_generator_param<RealType> param_;
};

}
