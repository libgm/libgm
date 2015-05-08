#ifndef LIBGM_MOMENT_GAUSSIAN_PARAM_HPP
#define LIBGM_MOMENT_GAUSSIAN_PARAM_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/serialization/eigen.hpp>

#include <algorithm>
#include <random>

#include <Eigen/Cholesky>

namespace libgm {

  // Forward declaration
  template <typename T> struct canonical_gaussian_param;

  /**
   * The parameters of a conditional multivariate normal (Gaussian) distribution
   * in the moment parameterization. The parameters represent a quadratic
   * function log p(x | y), where
   *
   * p(x | y) =
   *    1 / ((2*pi)^(m/2) det(cov)) *
   *    exp(-0.5 * (x - coef*y - mean)^T cov^{-1} (x - coef*y -mean) + c),
   *
   * where x an m-dimensional real vector, y is an n-dimensional real vector,
   * mean is the conditional mean, coef is an m x n matrix of coefficients,
   * and cov is a covariance matrix.
   *
   * \tparam T a real type for storing the parameters
   */
  template <typename T>
  struct moment_gaussian_param {
    // The underlying representation
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    //! The conditional mean.
    vec_type mean;

    //! The covariance matrix.
    mat_type cov;

    //! The coefficient matrix.
    mat_type coef;

    //! The log-multiplier.
    T lm;

    // Constructors and initialization
    //==========================================================================

    //! Constructs an empty moment Gaussian with the given log-multiplier.
    moment_gaussian_param(T lm = T(0))
      : lm(lm) { }

    //! Constructs a marginal Gaussian with the given sizes.
    moment_gaussian_param(std::size_t m, std::size_t n)
      : lm(0) {
      resize(m, n);
    }

    //! Constructs a marginal Gaussian with given parameters.
    moment_gaussian_param(const vec_type& mean,
                          const mat_type& cov,
                          T lm)
      : mean(mean), cov(cov), coef(mean.size(), 0), lm(lm) {
      check();
    }

    //! Constructs a conditional Gaussian with given parameters.
    moment_gaussian_param(const vec_type& mean,
                          const mat_type& cov,
                          const mat_type& coef,
                          T lm)
      : mean(mean), cov(cov), coef(coef), lm(lm) {
      check();
    }

    //! Copy constructor.
    moment_gaussian_param(const moment_gaussian_param& other) = default;

    //! Move constructor.
    moment_gaussian_param(moment_gaussian_param&& other) {
      swap(*this, other);
    }

    // Conversion from a canonical Gaussian representing a marginal distribution
    explicit moment_gaussian_param(const canonical_gaussian_param<T>& cg) {
      *this = cg;
    }

    //! Assignment operator.
    moment_gaussian_param& operator=(const moment_gaussian_param& other) {
      if (this != &other) {
        mean = other.mean;
        cov = other.cov;
        coef = other.coef;
        lm = other.lm;
      }
      return *this;
    }

    //! Move assignment operator.
    moment_gaussian_param& operator=(moment_gaussian_param&& other) {
      swap(*this, other);
      return *this;
    }

    //! Conversion from a canonical Gaussian.
    moment_gaussian_param& operator=(const canonical_gaussian_param<T>& cg) {
      Eigen::LLT<mat_type> chol(cg.lambda);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian: Cannot invert the precision matrix. "
          "Are you passing in a marginal canonical Gaussian distribution?"
        );
      }
      std::size_t n = cg.size();
      resize(n);
      mean = chol.solve(cg.eta);
      cov = chol.solve(mat_type::Identity(n, n));
      lm = cg.lm + T(0.5) * (n*log(two_pi<T>()) - logdet(chol)
                             + mean.dot(cg.eta));
      return *this;
    }

    //! Swaps the two sets of parameters.
    friend void swap(moment_gaussian_param& a, moment_gaussian_param& b) {
      a.mean.swap(b.mean);
      a.cov.swap(b.cov);
      a.coef.swap(b.coef);
      std::swap(a.lm, b.lm);
    }

    //! Serializes the parameters to an archive.
    void save(oarchive& ar) const {
      ar << mean << cov << coef << lm;
    }

    //! Deserializes the parameters from an archive.
    void load(iarchive& ar) {
      ar >> mean >> cov >> coef >> lm;
    }

    //! Resizes the parameters to the given head and tail vector size.
    void resize(std::size_t nhead, std::size_t ntail = 0) {
      mean.resize(nhead);
      cov.resize(nhead, nhead);
      coef.resize(nhead, ntail);
    }

    //! Initializes the parameters to 0 with given the head and tail size.
    void zero(std::size_t nhead, std::size_t ntail = 0) {
      mean.setZero(nhead);
      cov.setZero(nhead, nhead);
      coef.setZero(nhead, ntail);
      lm = T(0);
    }

    // Accessors and function evaluation
    //==========================================================================

    //! Returns the dimensionality of the function (head + tail).
    std::size_t size() const {
      return coef.rows() + coef.cols();
    }

    //! Returns the length of the head vector.
    std::size_t head_size() const {
      return coef.rows();
    }

    //! Returns the length of the tail vectgor.
    std::size_t tail_size() const {
      return coef.cols();
    }

    //! Returns true if the function represents a marginal distribution.
    bool is_marginal() const {
      return coef.cols() == 0;
    }

    //! Throws an exception if the matrix dimensions are inconsistent.
    void check() const {
      if (cov.rows() != cov.cols()) {
        throw std::invalid_argument(
          "The covariance matrix is not square."
        );
      }
      if (cov.rows() != mean.rows()) {
        throw std::invalid_argument(
          "The mean vector and covariance matrix have incompatible sizes."
        );
      }
      if (cov.rows() != coef.rows()) {
        throw std::invalid_argument(
          "The covariance and coefficient matrices are inconsistent."
        );
      }
    }

    //! Returns true if the two parameter structs are identical.
    friend bool operator==(const moment_gaussian_param& f,
                           const moment_gaussian_param& g) {
      return f.mean == g.mean && f.cov == g.cov && f.coef == g.coef
        && f.lm == g.lm;
    }

    //! Returns true if the two parameter structs are not identical.
    friend bool operator!=(const moment_gaussian_param& f,
                           const moment_gaussian_param& g) {
      return !(f == g);
    }

    //! Reorders the parameter according to the given index.
    moment_gaussian_param
    reorder(const matrix_index& head, const matrix_index& tail) const {
      assert(head_size() == head.size());
      assert(tail_size() == tail.size());
      moment_gaussian_param result;
      set(result.mean, subvec(mean, head));
      set(result.cov, submat(cov, head, head));
      set(result.coef, submat(coef, head, tail));
      result.lm = lm;
      return result;
    }

    //! Returns the log-value for the given head and tail vectors.
    T operator()(const vec_type& x, const vec_type& y = vec_type()) const {
      Eigen::LLT<mat_type> chol(cov);
      vec_type z = x - mean - coef * y;
      T log_norm = -std::log(two_pi<T>()) * head_size() - logdet(chol);
      T exponent = -z.transpose() * chol.solve(z);
      return T(0.5) * (log_norm + exponent) + lm;
    }

    /**
     * Returns the log-normalization constant of the distribution repesented
     * by these parameters.
     */
    T marginal() const {
      return lm;
    }

    /**
     * Returns the maximum value attained by the function.
     */
    T maximum() const {
      Eigen::LLT<mat_type> chol(cov);
      return T(0.5) * (-std::log(two_pi<T>())*head_size() - logdet(chol)) + lm;
    }

    /**
     * Returns the maximum value attained by the underlying factor and
     * the corresponding assignment, assuming that the function represent
     * a marginal distribution.
     */
    T maximum(vec_type& vec) const {
      assert(is_marginal());
      vec = mean;
      return maximum();
    }

    /**
     * Returns the entropy for the distribution represented by this Gaussian,
     * assuming that the function represents a marginal distribution.
     */
    T entropy() const {
      assert(is_marginal());
      Eigen::LLT<mat_type> chol(cov);
      return (size() * (std::log(two_pi<T>()) + T(1)) + logdet(chol)) / T(2);
    }

    /**
     * Returns the Kullback-Liebler divergence from p to q.
     * p and q must represent marginal distributions.
     */
    friend T kl_divergence(const moment_gaussian_param& p,
                           const moment_gaussian_param& q) {
      assert(p.is_marginal() && q.is_marginal());
      assert(p.head_size() == q.head_size());
      std::size_t m = p.head_size();
      Eigen::LLT<mat_type> cholp(p.cov);
      Eigen::LLT<mat_type> cholq(q.cov);
      auto identity = mat_type::Identity(m, m);
      T trace = (p.cov.array() * cholq.solve(identity).array()).sum();
      T means = (p.mean - q.mean).transpose() * cholq.solve(p.mean - q.mean);
      T logdets = -logdet(cholp) + logdet(cholq);
      return (trace + means + logdets - m) / T(2);
    }

    /**
     * Returns the maximum difference between the means, covariances,
     * and coefficients of two moment Gaussians.
     */
    friend T max_diff(const moment_gaussian_param& f,
                      const moment_gaussian_param& g) {
      T dmean = (f.mean - g.mean).array().abs().max();
      T dcov  = (f.cov - g.cov).array().abs().max();
      T dcoef = (f.coef - g.coef).array().abs().max();
      return std::max(std::max(dmean, dcov), dcoef);
    }

    // Sampling
    //==========================================================================

    /**
     * Draws a random sample from a marginal distribution.
     * This is only recommended for drawing a single sample. For multiple
     * samples, use gaussian_distribution.
     */
    template <typename Generator>
    vec_type sample(Generator& rng) const {
      return sample(rng, vec_type());
    }

    /**
     * Draws a random sample from a conditional distribution.
     * This is only recommended for drawing a single sample. For multiple
     * samples, use gaussian_distribution.
     */
    template <typename Generator>
    vec_type sample(Generator& rng, const vec_type& tail) const {
      Eigen::LLT<mat_type> chol(cov);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian::sample: Cannot compute the Cholesky decomposition"
        );
      }
      vec_type z(mean.size());
      std::normal_distribution<T> normal;
      for (std::size_t i = 0; i < z.size(); ++i) { z[i] = normal(rng); }
      return mean + chol.matrixL() * z + coef * tail;
    }

  }; // struct moment_gaussian_param

  /**
   * Prints the parameters to an output stream.
   */
  template <typename T>
  std::ostream&
  operator<<(std::ostream& out, const moment_gaussian_param<T>& p) {
    out << p.mean << std::endl
        << p.cov << std::endl
        << p.coef << std::endl
        << p.lm;
    return out;
  }

  // Join operations
  //============================================================================

  /**
   * A class that updates the log-multiplier of a moment Gaussian with constant.
   */
  template <typename T, typename Update>
  struct moment_gaussian_join_inplace {
    typedef moment_gaussian_param<T> param_type;

    /**
     * Constructs the operation for joining with a constant.
     */
    moment_gaussian_join_inplace() { }

    //! Joins a factor in-place with a constant.
    void operator()(param_type& result, T x) const {
      Update update;
      update(result.lm, x);
    }
  };

  /**
   * A class that multiplies two moment Gaussians.
   * We only support a special form of multiplication at the moment,
   * where one of the functions represents a marginal distribution,
   * whose head is disjoint from the head of the other factor.
   *
   * In the notation below, p(x_1, x_2) is one factor, and
   * q(x_3 | x_1, y) is another.
   * Then r(x_1, x_2, x_3 | y) or r(x_3, x_1, x_2 | y) is the result
   * (depending on the ordering of p and q).
   */
  template <typename T>
  struct moment_gaussian_multiplies {
    typedef moment_gaussian_param<T> param_type;

    matrix_index p1; //!< the indices of x1 in p
    matrix_index q1; //!< the indices of x1 in q
    matrix_index qy; //!< the indices of y in q
    matrix_index hp; //!< the indices of x1, x2 in h (always a block)
    matrix_index hq; //!< the indices of x3 in h (always a block)

    //! Default constructor (the caller must initialize indices manually).
    moment_gaussian_multiplies() { }

    //! Performs the multiplication operation
    void operator()(const param_type& f, const param_type& g,
                    param_type& h) const {
      // ensure that p is a marginal (consistent with multiplies_op)
      const param_type& p = f.is_marginal() ? f : g;
      const param_type& q = f.is_marginal() ? g : f;

      // define "all" indices for {x1, x2} in p and x3 in q
      matrix_index pall(0, p.head_size());
      matrix_index qall(0, q.head_size());

      // allocate and initialize the parameters
      h.resize(p.head_size() + q.head_size(), qy.size());
      h.coef.fill(T(0));
      h.lm = f.lm + g.lm;

      // extract the coefficients
      dynamic_matrix<T> a;
      set(a, submat(q.coef, qall, qy));
      h.coef.block(hq.start(), 0, a.rows(), a.cols()) = a;
      set(a, submat(q.coef, qall, q1));

      // compute the new conditional mean and covariance
      set(subvec(h.mean, hp), p.mean);
      set(submat(h.cov, hp, hp), p.cov);
      auto meanq = subvec(h.mean, hq).block();
      auto covqq = submat(h.cov, hq, hq).block();
      auto covpq = submat(h.cov, hp, hq).block();
      auto covqp = submat(h.cov, hq, hp).block();
      if (p1.contiguous()) {
        meanq.noalias() = q.mean + a * subvec(p.mean, p1).block();
        covqq.noalias() = q.cov + a * submat(p.cov, p1, p1).block() * a.transpose();
        covpq.noalias() = submat(p.cov, pall, p1).block() * a.transpose();
        covqp.noalias() = covpq.transpose();
      } else {
        meanq.noalias() = q.mean + a * subvec(p.mean, p1).plain();
        covqq.noalias() = q.cov + a * submat(p.cov, p1, p1).plain() * a.transpose();
        covpq.noalias() = submat(p.cov, pall, p1).plain() * a.transpose();
        covqp.noalias() = covpq.transpose();
      }
    }
  };

  // Aggregate operations
  //============================================================================

  /**
   * A class that computes the marginal or maximum of a moment Gaussian.
   */
  template <typename T>
  struct moment_gaussian_collapse {
    typedef moment_gaussian_param<T> param_type;

    /**
     * Constructs a collapse operator.
     * \param head_map the sequence of retained head indices
     * \param tail_map the sequence of retained tail indices
     * \param preserve_maximum if true, adjusts the log-multiplier so that the
     *        maximum value of the function is preserved
     */
    moment_gaussian_collapse(matrix_index&& head_map,
                             matrix_index&& tail_map,
                             bool preserve_maximum = false)
      : head_map(std::move(head_map)),
        tail_map(std::move(tail_map)),
        preserve_maximum(preserve_maximum) { }

    //! Performs the marginalization operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      set(h.mean, subvec(f.mean, head_map));
      set(h.cov,  submat(f.cov,  head_map, head_map));
      set(h.coef, submat(f.coef, head_map, tail_map));
      h.lm = f.lm;
      if (preserve_maximum) {
        h.lm += f.maximum() - h.maximum();
      }
    }

    matrix_index head_map;
    matrix_index tail_map;
    bool preserve_maximum;
  };

  // Conditioning operations
  //============================================================================

  /**
   * A class that can condition a marginal moment Gaussian distribution.
   * Specifically, if f represents p(x,y), this class computes p(x | y).
   */
  template <typename T>
  struct moment_gaussian_conditional {
    typedef moment_gaussian_param<T> param_type;
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    //! Constructs a conditioning operator.
    moment_gaussian_conditional(matrix_index&& x,
                                matrix_index&& y)
      : x(std::move(x)), y(std::move(y)) { }

    //! Performs the conditioning operation.
    void operator()(const param_type& f, param_type& h) {
      assert(f.is_marginal());

      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      set(cov_yy, submat(f.cov, y, y));
      set(sol_yx, submat(f.cov, y, x));
      chol_yy.compute(cov_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian::conditional: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);

      // compute the parameters of the conditional
      set(h.mean, subvec(f.mean, x));
      set(h.cov, submat(f.cov, x, x));
      if (y.contiguous()) {
        h.mean.noalias() -= sol_yx.transpose() * subvec(f.mean, y).block();
      } else {
        h.mean.noalias() -= sol_yx.transpose() * subvec(f.mean, y).plain();
      }
      if (x.contiguous() && y.contiguous()) {
        h.cov.noalias() -= submat(f.cov, x, y).block() * sol_yx;
      } else {
        h.cov.noalias() -= submat(f.cov, x, y).plain() * sol_yx;
      }
      h.coef.noalias() = sol_yx.transpose();
      h.lm = f.lm;
    }

    matrix_index x;
    matrix_index y;
    Eigen::LLT<mat_type> chol_yy;
    mat_type cov_yy;
    mat_type sol_yx;
  };

  /**
   * A class that restricts a moment Gaussian.
   *
   * There are two operations that this class supports:
   * 1) Marginal: given a distribution p(x, y | z), we partially restrict the
   *    head y = vec_y and fully restrict the tail z = vec_z. The result is a
   *    marginal distribution over x.
   * 2) Conditional: given a distribution p(z | x, y), we partially restrict
   *    the tail y = vec_y. The result is a conditional distribution with head
   *    z and tail x.
   */
  template <typename T>
  struct moment_gaussian_restrict {
    typedef moment_gaussian_param<T> param_type;
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    enum restrict_type { MARGINAL, CONDITIONAL };
    restrict_type type; //!< which version to perform

    matrix_index x; //!< the indices of the retained arguments in f
    matrix_index y; //!< the indices of the restricted arguments in f
    vec_type vec_y; //!< the assignment to the restricted arguments
    vec_type vec_z; //!< the assignment to the tail (for type = MARGINAL)

    //! Constructs a restrict operator of the given kind.
    moment_gaussian_restrict(restrict_type type)
      : type(type) { }

    //! Performs the restrict operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      if (type == MARGINAL) {
        restrict_marginal(f, h);
      } else if (type == CONDITIONAL) {
        restrict_conditional(f, h);
      } else {
        throw std::runtime_error("moment_gaussian: invalid restriction type");
      }
    }

    //! Performs a marginal restrict operation on f, storing the result in h.
    void restrict_marginal(const param_type f, param_type& h) {
      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      set(cov_yy, submat(f.cov, y, y));
      set(sol_yx, submat(f.cov, y, x));
      chol_yy.compute(cov_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian restrict: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);

      // some useful submatrices
      matrix_index z(0, f.tail_size());
      subvector<const vec_type> mean_x(f.mean, x);
      subvector<const vec_type> mean_y(f.mean, y);
      submatrix<const mat_type> coef_x(f.coef, x, z);
      submatrix<const mat_type> coef_y(f.coef, y, z);
      h.resize(x.size());

      // compute the residual over y (observation vec_y - the prediction)
      if (y.contiguous()) {
        res_y.noalias() = vec_y - mean_y.block() - coef_y.block() * vec_z;
      } else {
        res_y.noalias() = vec_y - mean_y.plain() - coef_y.plain() * vec_z;
      }

      // compute the mean: original mean + scaled residual
      set(h.mean, mean_x);
      if (x.contiguous()) {
        h.mean.noalias() += coef_x.block() * vec_z + sol_yx.transpose() * res_y;
      } else {
        h.mean.noalias() += coef_x.plain() * vec_z + sol_yx.transpose() * res_y;
      }

      // compute the covariance: original covariance - shrinking factor
      set(h.cov, submat(f.cov, x, x));
      if (x.contiguous() && y.contiguous()) {
        h.cov.noalias() -= submat(f.cov, x, y).block() * sol_yx;
      } else {
        h.cov.noalias() -= submat(f.cov, x, y).plain() * sol_yx;
      }

      // compute the log-multiplier
      h.lm = f.lm
        - T(0.5) * (y.size() * std::log(two_pi<T>()) + logdet(chol_yy))
        - T(0.5) * res_y.dot(chol_yy.solve(res_y));
    }

    //! Performs a conditional restrict operation on f, storing the result in h
    void restrict_conditional(const param_type& f, param_type& h) {
      matrix_index z(0, f.head_size());
      submatrix<const mat_type> coef_x(f.coef, z, x);
      submatrix<const mat_type> coef_y(f.coef, z, y);
      if (y.contiguous()) {
        h.mean.noalias() = f.mean + coef_y.block() * vec_y;
      } else {
       h.mean.noalias() = f.mean + coef_y.plain() + vec_y;
      }
      h.cov.noalias() = f.cov;
      set(h.coef, coef_x);
      h.lm = f.lm;
    }

    // temporary storage
    mat_type cov_yy;
    mat_type sol_yx;
    vec_type res_y;
    Eigen::LLT<mat_type> chol_yy;

  }; // struct moment_gaussian_restrict

  // TODO: KL project

} // namespace libgm

#endif
