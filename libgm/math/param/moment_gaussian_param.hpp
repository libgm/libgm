#ifndef LIBGM_MOMENT_GAUSSIAN_PARAM_HPP
#define LIBGM_MOMENT_GAUSSIAN_PARAM_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/range/integral.hpp>
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
  template <typename T = double>
  struct moment_gaussian_param {
    // The underlying representation
    typedef real_matrix<T> mat_type;
    typedef real_vector<T> vec_type;
    typedef std::vector<std::size_t> index_type;

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
      lm = cg.lm + T(0.5) * (n*std::log(two_pi<T>()) - logdet(chol)
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
    reorder(const index_type& head, const index_type& tail) const {
      assert(head_size() == head.size());
      assert(tail_size() == tail.size());
      moment_gaussian_param result;
      subvec(mean, head).eval_to(result.mean);
      submat(cov, head, head).eval_to(result.cov);
      submat(coef, head, tail).eval_to(result.coef);
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
      T dmean = (f.mean - g.mean).array().abs().maxCoeff();
      T dcov  = (f.cov - g.cov).array().abs().maxCoeff();
      T dcoef = (f.coef - g.coef).array().abs().maxCoeff();
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
   * In the notation below, p(x) is one factor, and q(y | x_1, z) is another,
   * where x_1 is a subset of x. Then h(x, y | z) or h(y, x | z) is the result
   * (depending on the ordering of p and q).
   */
  template <typename T>
  struct moment_gaussian_multiplies {
    typedef moment_gaussian_param<T> param_type;

    std::vector<std::size_t> p1; //!< the indices of x_1 in the head of p
    std::vector<std::size_t> q1; //!< the indices of x_1 in the tail of q
    std::vector<std::size_t> qz; //!< the indices of z   in the tail of q

    //! Default constructor (the caller must initialize indices manually).
    moment_gaussian_multiplies() { }

    //! Performs the multiplication operation
    void operator()(const param_type& f, const param_type& g,
                    param_type& h) const {
      // figure out which input factor is p and which q (see multiplies_op)
      const param_type& p = f.is_marginal() ? f : g;
      const param_type& q = f.is_marginal() ? g : f;

      // compute the position of x & y in h and their lengths
      std::size_t sx = f.is_marginal() ? 0 : f.head_size();
      std::size_t sy = f.is_marginal() ? f.head_size() : 0;
      std::size_t nx = p.head_size();
      std::size_t ny = q.head_size();

      // compute the "all" indices for x and y in p and q, respectively
      std::vector<std::size_t> px = range(0, p.head_size());
      std::vector<std::size_t> qy = range(0, q.head_size());

      // allocate and initialize the parameters
      h.resize(nx + ny, qz.size());
      h.coef.fill(T(0));
      h.lm = f.lm + g.lm;

      // extract the coefficients
      h.coef.block(sy, 0, ny, qz.size()) = submat(q.coef, qy, qz).ref();
      auto a = submat(q.coef, qy, q1);

      // create the blocks over the partitioned mean and covariance matrix
      auto mean_x = h.mean.segment(sx, nx).noalias();
      auto mean_y = h.mean.segment(sy, ny).noalias();
      auto cov_xx = h.cov.block(sx, sx, nx, nx).noalias();
      auto cov_yy = h.cov.block(sy, sy, ny, ny).noalias();
      auto cov_xy = h.cov.block(sx, sy, nx, ny).noalias();
      auto cov_yx = h.cov.block(sy, sx, ny, nx).noalias();

      // compute the new conditional mean and covariance
      mean_x = p.mean;
      mean_y = q.mean + a.ref() * subvec(p.mean, p1).ref();
      cov_xx = p.cov;
      cov_yy = q.cov + a.ref() * submat(p.cov, p1, p1).ref() * a.ref().transpose();
      cov_xy = submat(p.cov, px, p1).ref() * a.ref().transpose();
      cov_yx = h.cov.block(sx, sy, nx, ny).transpose();
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
    moment_gaussian_collapse(std::vector<std::size_t>&& head_map,
                             std::vector<std::size_t>&& tail_map,
                             bool preserve_maximum = false)
      : head_map(std::move(head_map)),
        tail_map(std::move(tail_map)),
        preserve_maximum(preserve_maximum) { }

    //! Performs the marginalization operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      subvec(f.mean, head_map).eval_to(h.mean);
      submat(f.cov,  head_map, head_map).eval_to(h.cov);
      submat(f.coef, head_map, tail_map).eval_to(h.coef);
      h.lm = f.lm;
      if (preserve_maximum) {
        h.lm += f.maximum() - h.maximum();
      }
    }

    std::vector<std::size_t> head_map;
    std::vector<std::size_t> tail_map;
    bool preserve_maximum;
  };

  // Conditioning operations
  //============================================================================

  /**
   * A class that can condition a marginal moment Gaussian distribution.
   * Specifically, if f represents p(x, y), this class computes p(x | y).
   */
  template <typename T>
  struct moment_gaussian_conditional {
    typedef moment_gaussian_param<T> param_type;
    typedef real_matrix<T> mat_type;
    typedef real_vector<T> vec_type;

    //! Constructs a conditioning operator.
    moment_gaussian_conditional(std::vector<std::size_t>&& x,
                                std::vector<std::size_t>&& y)
      : x(std::move(x)), y(std::move(y)) { }

    //! Performs the conditioning operation.
    void operator()(const param_type& f, param_type& h) {
      assert(f.is_marginal());

      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      submat(f.cov, y, y).eval_to(cov_yy);
      submat(f.cov, y, x).eval_to(sol_yx);
      chol_yy.compute(cov_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian::conditional: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);

      // compute the parameters of the conditional
      subvec(f.mean, x).eval_to(h.mean);
      submat(f.cov, x, x).eval_to(h.cov);
      h.mean.noalias() -= sol_yx.transpose() * subvec(f.mean, y).ref();
      h.cov.noalias() -= submat(f.cov, x, y).ref() * sol_yx;
      h.coef.noalias() = sol_yx.transpose();
      h.lm = f.lm;
    }

    std::vector<std::size_t> x;
    std::vector<std::size_t> y;
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
    typedef real_matrix<T> mat_type;
    typedef real_vector<T> vec_type;

    enum restrict_type { MARGINAL, CONDITIONAL };
    restrict_type type; //!< which version to perform

    std::vector<std::size_t> x; //!< the indices of the retained arguments in f
    std::vector<std::size_t> y; //!< the indices of the restricted arguments in f
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
      submat(f.cov, y, y).eval_to(cov_yy);
      submat(f.cov, y, x).eval_to(sol_yx);
      chol_yy.compute(cov_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian restrict: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);

      // some useful submatrices
      std::vector<std::size_t> z = range(0, f.tail_size());
      submatrix<const mat_type> coef_xz(f.coef, x, z);
      submatrix<const mat_type> coef_yz(f.coef, y, z);
      h.resize(x.size());

      // compute the residual over y (observation vec_y - the prediction)
      res_y.noalias() = vec_y - subvec(f.mean, y).ref() - coef_yz.ref() * vec_z;

      // compute the mean: original mean + scaled residual
      subvec(f.mean, x).eval_to(h.mean);
      h.mean.noalias() += coef_xz.ref() * vec_z + sol_yx.transpose() * res_y;

      // compute the covariance: original covariance - shrinking factor
      submat(f.cov, x, x).eval_to(h.cov);
      h.cov.noalias() -= submat(f.cov, x, y).ref() * sol_yx;

      // compute the log-multiplier
      h.lm = f.lm
        - T(0.5) * (y.size() * std::log(two_pi<T>()) + logdet(chol_yy))
        - T(0.5) * res_y.dot(chol_yy.solve(res_y));
    }

    //! Performs a conditional restrict operation on f, storing the result in h
    void restrict_conditional(const param_type& f, param_type& h) {
      std::vector<std::size_t> z = range(0, f.head_size());
      h.mean.noalias() = f.mean + submat(f.coef, z, y).ref() * vec_y;
      h.cov.noalias() = f.cov;
      h.lm = f.lm;
      submat(f.coef, z, x).eval_to(h.coef);
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
