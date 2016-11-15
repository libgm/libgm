#ifndef LIBGM_SOFTMAX_LL_HPP
#define LIBGM_SOFTMAX_LL_HPP

#include <libgm/datastructure/real_pair.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/eigen/hybrid.hpp>
#include <libgm/math/param/softmax_param.hpp>
#include <libgm/traits/is_sample_range.hpp>

#include <cmath>
#include <type_traits>

#include <Eigen/SparseCore>

namespace libgm {

  /**
   * A log-likelihood function of the softmax distribution and
   * its derivatives.
   *
   * The functions for computing slope and gradient accept a templated
   * argument of type Label, which can be either a std::size_t or an
   * expression that evaluates to an Eigen vector. In the former case,
   * the argument represents a single observation with the given label
   * index; in the latter case, the argument represents a distribution
   * over labels.
   *
   * \tparam T the real type representing the coefficients
   */
  template <typename T = double>
  class softmax_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The table of probabilities.
    typedef softmax_param<T> param_type;

    /**
     * Creates a log-likelihood function for a probability table with
     * the specified parameters.
     */
    explicit softmax_ll(const softmax_param<T>& f)
      : f(f) { }

    //! Returns the parameters of the log-likelihood function.
    const param_type& param() const {
      return f;
    }

    //! Returns the log-likelihood of the label for dense/sparse features.
    template <typename Derived>
    T value(std::size_t label, const Eigen::EigenBase<Derived>& x) const {
      return std::log(f(x)[label]);
    }

    //! Returns the log-likelihood of the label for sparse unit features.
    T value(std::size_t label, const std::vector<std::size_t>& x) const {
      return std::log(f(x)[label]);
    }

    //! Returns the log-likelihood of the specified data point.
    T value(const hybrid_vector<T>& index) const {
      return std::log(f(index.real())[index.uint()[0]]);
    }

    /**
     * Returns a pair consisting of the log-likelihood of a datapoint
     * specified as a label and a dense Eigen feature vector, as well as
     * the slope of the log-likelihood along the given direction.
     */
    template <typename Label>
    real_pair<T> value_slope(const Label& label, const dense_vector<T>& x,
                             const softmax_param<T>& dir) const {
      std::pair<T, dense_vector<T> > d = slope_delta(label, x);
      T wslope = dir.weight().cwiseProduct(d.second * x.transpose()).sum();
      return {d.first, dir.bias().dot(d.second) + wslope};
    }

    /**
     * Returns a pair consisting of the log-likelihood of a datapoint
     * specified as a label and a sparse Eigen feature vector, as well as
     * the slope of the log-likelihood along the given direction.
     */
    template <typename Label>
    real_pair<T> value_slope(const Label& label,
                             const Eigen::SparseVector<T>& x,
                             const softmax_param<T>& dir) const {
      std::pair<T, dense_vector<T> > d = slope_delta(label, x);
      real_pair<T> result(d.first, dir.bias().dot(d.second));
      for (typename Eigen::SparseVector<T>::InnerIterator it(x); it; ++it) {
        result.second += dir.weight().col(it.index()).dot(d.second)*it.value();
      }
      return result;
    }

    /**
     * Returns a pair consisting of the log-likelihood of a datapoint
     * specified as a label and a sparse unit feature vector, as well as
     * the slope of the log-likelihood along the given direction.
     */
    template <typename Label>
    real_pair<T> value_slope(const Label& label,
                             const std::vector<std::size_t>& x,
                             const softmax_param<T>& dir) const {
      std::pair<T, dense_vector<T> > d = slope_delta(label, x);
      real_pair<T> result(d.first, dir.bias().dot(d.second));
      for (std::size_t i : x) {
        result.second += dir.weight().col(i).dot(d.second);
      }
      return result;
    }

    /**
     * Returns a pair consisting of the log-likelihood of a datapoint
     * specified as a hybrid_vector, as well as
     * the slope of the log-likelihood along the given direction.
     */
    real_pair<T> value_slope(const hybrid_vector<T>& index,
                             const softmax_param<T>& dir) const {
      return value_slope(index.uint()[0], index.real(), dir);
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a dense Eigen feature vector.
     */
    template <typename Label>
    void add_gradient(const Label& label, const dense_vector<T>& x, T w,
                      softmax_param<T>& g) const {
      dense_vector<T> p = gradient_delta(label, x, w);
      g.weight().noalias() += p * x.transpose();
      g.bias() += p;
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a sparse Eigen feature vector.
     */
    template <typename Label>
    void add_gradient(const Label& label, const Eigen::SparseVector<T>& x, T w,
                      softmax_param<T>& g) const {
      dense_vector<T> p = gradient_delta(label, x, w);
      for (typename Eigen::SparseVector<T>::InnerIterator it(x); it; ++it) {
        g.weight().col(it.index()) += p * it.value();
      }
      g.bias() += p;
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a sparse unit feature vector.
     */
    template <typename Label>
    void add_gradient(const Label& label,
                      const std::vector<std::size_t>& x, T w,
                      softmax_param<T>& g) const {
      dense_vector<T> p = gradient_delta(label, x, w);
      for (std::size_t i : x) { g.weight().col(i) += p; }
      g.bias() += p;
    }

    /**
     * Adds gradient of the log-likelihood to g for a datapoint
     * specified as hybrid_vector.
     */
    void add_gradient(const hybrid_vector<T>& index, T w,
                     softmax_param<T>& g) const {
      add_gradient(index.uint()[0], index.real(), w, g);
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a dense Eigen feature vector.
     */
    void add_hessian_diag(const dense_vector<T>& x, T w,
                          softmax_param<T>& h) const {
      dense_vector<T> v = hessian_delta(x, w);
      h.weight().noalias() += v * x.cwiseProduct(x).transpose();
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a sparse Eigen feature vector.
     */
    void add_hessian_diag(const Eigen::SparseVector<T>& x, T w,
                          softmax_param<T>& h) const {
      dense_vector<T> v = hessian_delta(x, w);
      for (typename Eigen::SparseVector<T>::InnerIterator it(x); it; ++it) {
        h.weight().col(it.index()) += v * (it.value() * it.value());
      }
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a sparse feature vector with unit values.
     */
    void add_hessian_diag(const std::vector<std::size_t>& x, T w,
                          softmax_param<T>& h) const {
      dense_vector<T> v = hessian_delta(x, w);
      for (std::size_t i : x) { h.weight().col(i) += v; }
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a hybrid index.
     */
    void add_hessian_diag(const hybrid_vector<T>& x, T w,
                          softmax_param<T>& h) const {
      add_hessian_diag(x.real(), w, h);
    }

  private:
    template <typename Features>
    std::pair<T, dense_vector<T> >
    slope_delta(std::size_t label, const Features& x) const {
      dense_vector<T> p = f(x);
      T value = std::log(p[label]);
      p[label] -= T(1);
      p = -p;
      return {value, p};
    }

    template <typename Features>
    std::pair<T, dense_vector<T> >
    slope_delta(const Eigen::Ref<const dense_vector<T> >& plabel,
                const Features& x) const {
      dense_vector<T> p = f(x);
      T value = p.unaryExpr(logarithm<T>()).dot(plabel);
      p -= plabel;
      p = -p;
      return {value, p};
    }

    template <typename Features>
    dense_vector<T>
    gradient_delta(std::size_t label, const Features& x, T w) const {
      dense_vector<T> p = f(x);
      p[label] -= T(1);
      p *= -w;
      return p;
    }

    template <typename Features>
    dense_vector<T>
    gradient_delta(const Eigen::Ref<const dense_vector<T> >& plabel,
                   const Features& x, T w) const {
      dense_vector<T> p = f(x);
      p -= plabel;
      p *= -w;
      return p;
    }

    template <typename Features>
    dense_vector<T> hessian_delta(const Features& x, T w) const {
      dense_vector<T> v = f(x);
      v -= v.cwiseProduct(v);
      v *= -w;
      return v;
    }

    //! The parameters at which we evaluate the log-likelihood derivatives.
    const softmax_param<T>& f;

  }; // class softmax_ll

} // namespace libgm

#endif
