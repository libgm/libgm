#ifndef LIBGM_MATRIX_GENERATOR_HPP
#define LIBGM_MATRIX_GENERATOR_HPP

#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/generator/diagonal_generator_param.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

namespace libgm {

  /**
   * A generator that returns a matrix filled with independent draws from
   * a random number distribution.
   *
   * \tparam Distribution
   *         A class that models the RandomNumberDistribution class.
   */
  template <typename Distribution>
  class matrix_generator {
    // ParameterGenerator types
    using real_type = typename Distribution::real_type;
    using result_type = dense_matrix<real_type>;
    using shape_type = std::pair<std::size_t, std::size_t>;
    using param_type = typename Distribution::param_type;

    /**
     * Constructs a matrix_generator, passing the parameters down to
     * the distribution.
     */
    template <typename... Arg>
    matrix_generator(Arg&&... arg)
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
    friend std::ostream&
    operator<<(std::ostream& out, const matrix_generator& g) {
      out << "matrix_generator(" << g.distribution_ << ")";
      return out;
    }

    /**
     * Generates a matrix using the stored random number distribution.
     */
    template <typename Generator>
    dense_matrix<real_type>
    operator()(std::size_t rows, std::size_t cols, Generator& g) const {
      dense_matrix<real_type> r(rows, cols);
      std::generate(r.data(), r.data() + r.size(),
                    std::bind(distribution_, std::ref(g)));
      return r;
    }

    /**
     * Generates a matrix using the stored random number distribution.
     */
    template <typename Generator>
    dense_matrix<real_type>
    operator()(std::pair<std::size_t, std::size_t> shape, Generator& g) const {
      return operator()(shape.first, shape.second);
    }

  private:
    Distribution distribution_;
  };

  /**
   * A matrix_generator that draws the matrix elements from a uniform
   * distribution.
   * \relates matrix_generator
   */
  template <typename RealType = double>
  using uniform_matrix_generator =
    matrix_generator<std::uniform_distribution<RealType> >;

  /**
   * A matrix_generator that returns a matrix whose parameters are drawn from a
   * gamma distribution.
   *
   * In the special case, when the second parameter of the gamma distribution is
   * 1, and the matrix is normalized to sum to 1, this results in a draw from
   * a dirichlet distribution.
   *
   * \relates matrix_generator
   */
  template <typename RealType = double>
  using dirichlet_matrix_generator =
    matrix_generator<std::gamma_distribution<RealType> >;

  /**
   * A generator that draws a random matrix that is a sum of a fixed base
   * number and a constant diagonal matrix, whose value is drawn from
   * a uniform distribution. This can be used to generate associative
   * factors and ising factors.
   */
  template <typename RealType = double>
  class diagonal_matrix_generator {
  public:
    // ParameterGenerator types
    using real_type = RealType;
    using result_type = dense_matrix<RealType>;
    using shape_type = std::size_t;
    using param_type = diagonal_generator_param<RealType>;

    /**
     * Constructs a diagonal_matrix_generator, passing the arguments down to
     * the diagonal_generator_param struct.
     */
    template <typename... Arg>
    diagonal_matrix_generator(Arg&&... arg)
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
    operator<<(std::ostream& out, const matrix_generator& g) {
      out << "diagonal_matrix_generator(" << g.param_ << ")";
      return out;
    }

    /**
     * Generates a matrix using the stored random number distribution.
     */
    template <typename Generator>
    dense_matrix<real_type> operator()(std::size_t n, Generator& g) const {
      std::uniform_real_distribution<RealType> offset(param_.lower,
                                                      param_.upper);
      return dense_matrix<RealType>::Constant(n, n, param_.base) +
        dense_matrix<RealType>::Identity(n, n, offset(g));
    }

  private:
    diagonal_generator_param<RealType> param_;

} // namespace libgm

#endif
