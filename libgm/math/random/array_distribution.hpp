#ifndef LIBGM_ARRAY_DISTRIBUTION_HPP
#define LIBGM_ARRAY_DISTRIBUTION_HPP

#include <libgm/datastructure/finite_index.hpp>
#include <libgm/math/log_tag.hpp>

#include <random>
#include <stdexcept>

#include <Eigen/Core>

namespace libgm {

  /**
   * A categorical distribution over multiple arguments,
   * whose probabilities are represented by an Eigen array.
   */
  template <typename T, size_t N> class array_distribution { };  

  /**
   * A categorical distribution over a single argument,
   * whose probabilities are represented by an Eigen array.
   */
  template <typename T>
  class array_distribution<T, 1> {
  public:
    //! The underlying parameter type.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> param_type;

    //! The type representing the sample.
    typedef size_t result_type;

    //! Constructor for a distribution in the probability space.
    explicit array_distribution(const param_type& p)
      : psum_(p) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Constructor for a distribution in the log space.
    array_distribution(const param_type& p, log_tag)
      : psum_(exp(p)) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    size_t operator()(Generator& rng) const {
      const T* begin = psum_.data();
      T p = std::uniform_real_distribution<T>()(rng);
      size_t i  = std::upper_bound(begin, begin + psum_.size(), p) - begin;
      if (i < psum_.size()) {
        return i;
      } else {
        throw std::invalid_argument("The probabilities are less than 1");
      }
    }

  private:
    //! Partial sums.
    param_type psum_;

  }; // class array_distribution<T, 1>

  /**
   * A categorical distribution over two arguments,
   * whose probabilities are represented by an Eigen array.
   */
  template <typename T>
  class array_distribution<T, 2> {
  public:
    //! The underlying parameter type.
    typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> param_type;
    
    //! The type representing the sample.
    typedef finite_index result_type;

    //! The type representing the assignment to the tail.
    typedef size_t tail_type;

    //! Constructor for a distribution in the probability space.
    explicit array_distribution(const param_type& p)
      : psum_(p) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Constructor for a distribution in the log space.
    array_distribution(const param_type& p, log_tag)
      : psum_(exp(p)) {
      std::partial_sum(psum_.data(), psum_.data() + psum_.size(), psum_.data());
    }

    //! Draws a random sample from a marginal distribution.
    template <typename Generator>
    finite_index operator()(Generator& rng) const {
      const T* begin = psum_.data();
      T p = std::uniform_real_distribution<T>()(rng);
      size_t i  = std::upper_bound(begin, begin + psum_.size(), p) - begin;
      if (i < psum_.size()) {
        return { i % psum_.rows(), i / psum_.rows() };
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

    //! Draws a random sample from a conditional distribution.
    template <typename Generator>
    size_t operator()(Generator& rng, size_t tail) const {
      const T* begin = psum_.data() + tail * psum_.rows();
      T p = std::uniform_real_distribution<T>()(rng);
      if (tail > 0) { p += *(begin-1); }
      size_t i = std::upper_bound(begin, begin + psum_.rows(), p) - begin;
      if (i < psum_.rows()) {
        return i;
      } else {
        throw std::invalid_argument("The total probability is less than 1");
      }
    }

  private:
    //! Partial sums.
    param_type psum_;

  }; // class array_distribution<T, 2>

} // namespace libgm

#endif
