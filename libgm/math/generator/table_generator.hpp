#ifndef LIBGM_TABLE_GENERATOR_HPP
#define LIBGM_TABLE_GENERATOR_HPP

#include <libgm/datastructure/table.hpp>
#include <libgm/datastructure/uint_vector.hpp>

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
  class table_generator {
    // ParameterGenerator types
    using real_type = typename Distribution::real_type;
    using result_type = table<real_type>;
    using shape_type = uint_vector;
    using param_type = typename Distribution::param_type;

    /**
     * Constructs a table_generator, passing the parameters down to
     * the distribution.
     */
    template <typename... Arg>
    table_generator(Arg&&... arg)
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
    operator<<(std::ostream& out, const table_generator& g) {
      out << "table_generator(" << g.distribution_ << ")";
      return out;
    }

    /**
     * Generates a table using the stored random number distribution.
     */
    template <typename Generator>
    table<real_type> operator()(const uint_vector& shape, Generator& g) const {
      table<real_type> r(shape);
      std::generate(r.begin(), r.end(), std::bind(distribution_, std::ref(g)));
      return r;
    }

  private:
    Distribution distribution_;
  };

  /**
   * A table_generator that draws the table elements from a uniform
   * distribution.
   *
   * \relates table_generator
   */
  template <typename RealType = double>
  using uniform_table_generator =
    table_generator<std::uniform_distribution<RealType> >;

  /**
   * A table_generator that returns a table whose parameters are drawn from a
   * gamma distribution.
   *
   * In the special case, when the second parameter of the gamma distribution is
   * 1, and the table is normalized to sum to 1, this results in a draw from
   * a dirichlet distribution.
   *
   * \relates table_generator
   */
  template <typename RealType = double>
  using dirichlet_table_generator =
    table_generator<std::gamma_distribution<RealType> >;

  /**
   * A generator that draws a random table that is a sum of a fixed base
   * number and a constant diagonal matrix, whose value is drawn from
   * a uniform distribution. This can be used to generate associative
   * factors and ising factors.
   */
  template <typename Distribution>
  class diagonal_table_generator {
    // ParameterGenerator types
    using real_type = RealType;
    using result_type = table<RealType>;
    using shape_type = std::pair<std::size_t, std::size_t>;
    using param_type = diagonal_generator_param<RealType>;

    /**
     * Constructs a diagonal_table_generator, passing the arguments down to
     * the diagonal_generator_param struct.
     */
    template <typename... Arg>
    diagonal_table_generator(Arg&&... arg)
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
    operator<<(std::ostream& out, const table_generator& g) {
      out << "diagonal_table_generator(" << g.param_ << ")";
      return out;
    }

    /**
     * Generates a table using the stored random number distribution.
     */
    template <typename Generator>
    table<RealType>
    operator()(std::size_t arity, std::size_t n, Generator& g) const {
      RealType x = std::uniform_real_distribution<RealType>(
        param_.lower, param_.upper
      )(rng);
      table<real_type> r(uint_vector(arity, n), param_.base);
      uint_vector index;
      for (std::size_t k = 0; k < n; ++k) {
        index.assign(arity, k);
        f.param(index) += x;
      }
      return r;
    }

  private:
    diagonal_generator_param<RealType> param_;

  }; // class diagonal_table_generator

} // namespace libgm

#endif
