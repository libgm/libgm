#ifndef LIBGM_VECTOR_GENERATOR_HPP
#define LIBGM_VECTOR_GENERATOR_HPP

#include <libgm/math/eigen/dense.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

namespace libgm {

  /**
   * A generator that returns a vector filled with independent draws from
   * a random number distribution.
   *
   * \tparam Distribution
   *         A class that models the RandomNumberDistribution class.
   */
  template <typename Distribution>
  class vector_generator {
    // ParameterGenerator types
    using real_type = typename Distribution::real_type;
    using result_type = dense_vector<real_type>;
    using shape_type = std::size_t;
    using param_type = typename Distribution::param_type;

    /**
     * Constructs a vector_generator, passing the parameters down to
     * the distribution.
     */
    template <typename... Arg>
    vector_generator(Arg&&... arg)
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
    operator<<(std::ostream& out, const vector_generator& g) {
      out << "vector_generator(" << g.distribution_ << ")";
      return out;
    }

    /**
     * Generates a vector using the stored random number distribution.
     */
    template <typename Generator>
    dense_vector<real_type> operator()(std::size_t length, Generator& g) const {
      dense_vector<real_type> r(length);
      std::generate(r.data(), r.data() + r.size(),
                    std::bind(distribution_, std::ref(g)));
      return r;
    }

  private:
    Distribution distribution_;
  };

  /**
   * A vector_generator that draws the vector elements from a uniform
   * distribution.
   * \relates vector_generator
   */
  template <typename RealType = double>
  using uniform_vector_generator =
    vector_generator<std::uniform_distribution<RealType> >;

  /**
   * A vector_generator that returns a vector whose parameters are drawn from a
   * gamma distribution.
   *
   * In the special case, when the second parameter of the gamma distribution is
   * 1, and the vector is normalized to sum to 1, this results in a draw from
   * a dirichlet distribution.
   *
   * \relates vector_generator
   */
  template <typename RealType = double>
  using dirichlet_vector_generator =
    vector_generator<std::gamma_distribution<RealType> >;

} // namespace libgm

#endif
