#ifndef LIBGM_MULTIVARIATE_CATEGORICAL_DISTRIBUTION_HPP
#define LIBGM_MULTIVARIATE_CATEGORICAL_DISTRIBUTION_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/table.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/composition.hpp>
#include <libgm/math/tags.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace libgm {

  /**
   * A categorical distribution over multiple arguments,
   * whose probabilities are represented by a table.
   */
  template <typename T = double>
  class multivariate_categorical_distribution {
  public:

    //! The type representing the parameters of the distribution.
    typedef table<T> param_type;

    //! The type representing the sample.
    typedef uint_vector result_type;

    //! The type representing the assignment to the tail.
    typedef uint_vector tail_type;

    //! Constructor for a distribution in the probability space.
    multivariate_categorical_distribution(const table<T>& p, std::size_t ntail)
      : ntail_(ntail) {
      init(p, identity());
    }

    //! Constructor for a distribution in the log space.
    multivariate_categorical_distribution(const table<T>& lp, std::size_t ntail,
                                          log_tag)
      : ntail_(ntail) {
      init(lp, exponent<T>());
    }

    /**
     * Draws a random sample from a marginal distribution.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    uint_vector operator()(Generator& rng) const {
      return operator()(rng, uint_vector());
    }

    /**
     * Draws a random sample from a distribution conditioned on the given
     * assignment to tail dimensions.
     *
     * \throw std::out_of_range
     *        may be thrown if the distribution is not normalized
     */
    template <typename Generator>
    uint_vector operator()(Generator& rng, const uint_vector& tail) const {
      assert(tail.size() == ntail_);

      // compute the range of elements we search over
      std::size_t d = psum_.arity() - ntail_;
      const T* begin = psum_.begin() + psum_.offset().linear(tail, d);
      const T* end = begin + psum_.offset().multiplier(d);

      // compute the probability we search for
      T p = std::uniform_real_distribution<T>()(rng);
      if (begin > psum_.begin()) { p += *(begin-1); }

      // compute the index
      uint_vector index;
      psum_.offset().vector(std::upper_bound(begin, end, p) - begin, d, index);
      return index;
    }

  private:
    /**
     * Reorders the elements of the input table such that the elements for each
     * tail assignment are grouped together, and computes the partial sums,
     * transforming the elements using the specified unary operation.
     */
    template <typename UnaryOp>
    void init(const table<T>& param, UnaryOp unary_op) {
      std::size_t n = param.arity();
      assert(ntail_ < n);
      assert(!param.empty());
      if (ntail_ == 0) {
        psum_ = param;
      } else {
        param.reorder(concat(range(ntail_, n), range(0, ntail_)), psum_);
      }
      psum_[0] = unary_op(psum_[0]);
      std::partial_sum(psum_.begin(), psum_.end(), psum_.begin(),
                       compose_right(std::plus<T>(), unary_op));
    }

    /**
     * The number of tail dimensions.
     */
    std::size_t ntail_;

    /**
     * The table of partial sums of probabilities. The data is stored
     * in such a way that all the elements for one tail assignment are
     * stored contiguously (i.e., the reverse order than usual).
     */
    table<T> psum_;


  }; // class multivariate_categorical_distribution

} // namespace libgm

#endif
