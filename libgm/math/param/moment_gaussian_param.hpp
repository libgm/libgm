#ifndef LIBGM_MOMENT_GAUSSIAN_PARAM_HPP
#define LIBGM_MOMENT_GAUSSIAN_PARAM_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/range/index_range.hpp>
#include <libgm/range/index_range_complement.hpp>
#include <libgm/serialization/eigen.hpp>

#include <algorithm>
#include <random>

#include <Eigen/Cholesky>

namespace libgm {

  // Forward declaration
  template <typename RealType> struct canonical_gaussian_param;

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
  template <typename RealType = double>
  struct moment_gaussian_param {

    // Public types
    //--------------------------------------------------------------------------

    //! The type storing the parameters.
    typedef RealType value_type;

    //! The type of the LLT Cholesky decomposition object.
    typedef Eigen::LLT<dense_matrix<RealType> > cholesky_type;

    //! The struct storing the temporaries for conditioning computation.
    struct conditional_workspace {
      cholesky_type chol_yy;
      dense_matrix<RealType> sol_yx;
    };

    //! The struct storing the temporaries for restrict computation.
    struct restrict_workspace {
      cholesky_type chol_yy;
      dense_matrix<RealType> cov_yy;
      dense_matrix<RealType> sol_yx;
      dense_vector<RealType> res_y;
    };

    // The parameters
    //--------------------------------------------------------------------------

    //! The conditional mean.
    dense_vector<RealType> mean;

    //! The covariance matrix.
    dense_matrix<RealType> cov;

    //! The coefficient matrix.
    dense_matrix<RealType> coef;

    //! The log-multiplier.
    RealType lm;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    //! Constructs an empty moment Gaussian with the given log-multiplier.
    moment_gaussian_param(RealType lm = RealType(0))
      : lm(lm) { }

    //! Constructs a marginal Gaussian with the given sizes.
    moment_gaussian_param(std::size_t m, std::size_t n)
      : lm(0) {
      resize(m, n);
    }

    //! Constructs a marginal Gaussian with given parameters.
    moment_gaussian_param(const dense_vector<RealType>& mean,
                          const dense_matrix<RealType>& cov,
                          RealType lm)
      : mean(mean), cov(cov), coef(mean.size(), 0), lm(lm) {
      check();
    }

    //! Constructs a conditional Gaussian with given parameters.
    moment_gaussian_param(const dense_vector<RealType>& mean,
                          const dense_matrix<RealType>& cov,
                          const dense_matrix<RealType>& coef,
                          RealType lm)
      : mean(mean), cov(cov), coef(coef), lm(lm) {
      check();
    }

    // Conversion from a canonical Gaussian representing a marginal distribution
    explicit moment_gaussian_param(
        const canonical_gaussian_param<RealType>& cg) {
      *this = cg;
    }

    //! Conversion from a canonical Gaussian.
    moment_gaussian_param&
    operator=(const canonical_gaussian_param<RealType>& cg) {
      cholesky_type chol(cg.lambda);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian: Cannot invert the precision matrix. "
          "Are you passing in a marginal canonical Gaussian distribution?"
        );
      }
      std::size_t n = cg.size();
      resize(n);
      mean = chol.solve(cg.eta);
      cov = chol.solve(dense_matrix<RealType>::Identity(n, n));
      lm = cg.lm + (n * std::log(two_pi<RealType>()) - logdet(chol)
                    + mean.dot(cg.eta)) / RealType(2);
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

    //! Resizes the parameters to the given ehad nad tail vector size.
    void resize(std::pair<std::size_t, std::size_t> n) {
      resize(n.first, n.second);
    }

    //! Initializes the parameters to 0 with given the head and tail size.
    void zero(std::size_t nhead, std::size_t ntail = 0) {
      mean.setZero(nhead);
      cov.setZero(nhead, nhead);
      coef.setZero(nhead, ntail);
      lm = RealType(0);
    }

    // Accessors and function evaluation
    //--------------------------------------------------------------------------

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

    //! Throws std::logic_error if the matrix dimensions are invalid.
    void check_size(std::pair<std::size_t, std::size_t> n) const {
      if (cov.rows() != cov.cols()) {
        throw std::logic_error("The covariance matrix is not square.");
      }
      if (mean.rows() != n.first) {
        throw std::logic_error("Invalid size of the mean vector.");
      }
      if (cov.rows() != n.first) {
        throw std::logic_error("Invalid size of the covariance matrix.");
      }
      if (coef.rows() != n.first || coef.cols() != n.second) {
        throw std::logic_error("Invalid shape of the coefficient matrix.");
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

    // Accessors and function evaluation
    //--------------------------------------------------------------------------

    /**
     * Computes the log-value for the given head and tail vectors.
     */
    RealType operator()(const dense_vector<RealType>& x,
                        const dense_vector<RealType>& y =
                          dense_vector<RealType>()) const {
      cholesky_type chol(cov);
      dense_vector<RealType> z = x - mean - coef * y;
      RealType log_norm = -std::log(two_pi<RealType>()) * head_size() - logdet(chol);
      RealType exponent = -z.transpose() * chol.solve(z);
      return RealType(0.5) * (log_norm + exponent) + lm;
    }

    /**
     * Returns the marginal (in the log space) of the function represented
     * by these parameters.
     */
    RealType marginal() const {
      return lm;
    }

    /**
     * Returns the maximum value (in the log space) attained by the function
     * represented by these parameters. The function must be normalizable.
     */
    RealType maximum() const {
      cholesky_type chol(cov);
      return (-std::log(two_pi<RealType>()) * head_size() - logdet(chol))
        / RealType(2) + lm;
    }

    /**
     * Returns the maximum value (in the log space) attained by the function
     * represented by these parameters and stores the corresponding value to
     * a vector. The function must be normalizable.
     */
    RealType maximum(dense_vector<RealType>& vec) const {
      assert(is_marginal());
      vec = mean;
      return maximum();
    }

    /**
     * Returns the entropy for the marginal distribution represented by these
     * parameters.
     */
    RealType entropy() const {
      assert(is_marginal());
      cholesky_type chol(cov);
      return (size() * (std::log(two_pi<RealType>()) + RealType(1))
              + logdet(chol)) / RealType(2);
    }

    /**
     * Returns the Kullback-Leibler divergence from p to q.
     */
    friend RealType kl_divergence(const moment_gaussian_param& p,
                                  const moment_gaussian_param& q) {
      assert(p.is_marginal() && q.is_marginal());
      assert(p.head_size() == q.head_size());
      std::size_t m = p.head_size();
      cholesky_type cholp(p.cov);
      cholesky_type cholq(q.cov);
      auto identity = dense_matrix<RealType>::Identity(m, m);
      RealType trace = (p.cov.array() * cholq.solve(identity).array()).sum();
      RealType means = (p.mean - q.mean).transpose() * cholq.solve(p.mean - q.mean);
      RealType logdets = -logdet(cholp) + logdet(cholq);
      return (trace + means + logdets - m) / RealType(2);
    }

    /**
     * Returns the maximum difference between the means, covariances,
     * and coefficients of two moment Gaussians.
     */
    friend RealType max_diff(const moment_gaussian_param& f,
                             const moment_gaussian_param& g) {
      RealType dmean = (f.mean - g.mean).array().abs().maxCoeff();
      RealType dcov  = (f.cov - g.cov).array().abs().maxCoeff();
      RealType dcoef = (f.coef - g.coef).array().abs().maxCoeff();
      return std::max({dmean, dcov, dcoef});
    }

    // Sampling
    //--------------------------------------------------------------------------

    /**
     * Draws a random sample from a marginal distribution.
     * This is only recommended for drawing a single sample. For multiple
     * samples, use multivariate_normal_distribution.
     */
    template <typename Generator>
    dense_vector<RealType> sample(Generator& rng) const {
      return sample(rng, dense_vector<RealType>());
    }

    /**
     * Draws a random sample from a conditional distribution.
     * This is only recommended for drawing a single sample. For multiple
     * samples, use multivariate_normal_distribution.
     */
    template <typename Generator>
    dense_vector<RealType>
    sample(Generator& rng, const dense_vector<RealType>& tail) const {
      cholesky_type chol(cov);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian::sample: Cannot compute the Cholesky decomposition"
        );
      }
      dense_vector<RealType> z(mean.size());
      std::normal_distribution<RealType> normal;
      for (std::ptrdiff_t i = 0; i < z.size(); ++i) {
        z[i] = normal(rng);
      }
      return mean + chol.matrixL() * z + coef * tail;
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Transforms the parameters with the given vector and scalar operations.
     */
    template <typename VectorOp, typename ScalarOp, typename OtherT>
    void transform(VectorOp vector_op, ScalarOp scalar_op,
                   moment_gaussian_param<OtherT>& result) const {
      result.mean = vector_op(mean);
      result.cov  = vector_op(cov);
      result.coef = vector_op(coef);
      result.lm   = scalar_op(lm);
    }

    /**
     * Multiplies two moment_gaussians when (a range of) the head of one operand
     * matches (a range of) the tail of the other operand. The ordering of the
     * operands is specified via the forward flag.
     */
    template <typename HeadIt, typename TailIt>
    friend void multiply_head_tail(const moment_gaussian_param& f,
                                   const moment_gaussian_param& g,
                                   index_range<HeadIt> f_head,
                                   index_range<TailIt> g_tail,
                                   bool forward,
                                   moment_gaussian_param& r) {
      assert(f_head.size() == g_tail.size());

      // compute the positions of the head and tail of f and g in the output
      std::size_t n = f_head.size();
      span x(forward ? 0 : g.head_size(), f.head_size());     // head of f
      span y(forward ? f.head_size() : 0, g.head_size());     // head of g
      span v(forward ? 0 : g.tail_size() - n, f.tail_size()); // tail of f
      span w(forward ? f.tail_size() : 0, g.tail_size() - n); // tail of g

      // compute the result
      r.resize(f.head_size() + g.head_size(),
               f.tail_size() + g.tail_size() - n);
      dense_matrix<RealType> coef = subcols(g.coef, g_tail); // used frequently
      subvec(r.mean, x) = f.mean;
      subvec(r.mean, y).noalias() = g.mean + coef * subvec(f.mean, f_head);
      submat(r.cov, x, x) = f.cov;
      submat(r.cov, x, y).noalias() = subcols(f.cov, f_head) * coef.transpose();
      submat(r.cov, y, x) = submat(r.cov, x, y).transpose();
      submat(r.cov, y, y).noalias() =
        g.cov + coef * submat(f.cov, f_head, f_head) * coef.transpose();
      submat(r.coef, x, v) = f.coef;
      submat(r.coef, x, w) = dense_matrix<RealType>::Zero(x.size(), w.size());
      submat(r.coef, y, v).noalias() = coef * subrows(f.coef, f_head);
      submat(r.coef, y, w) = subcols(g.coef, complement(g_tail, g.tail_size()));
      r.lm = f.lm + g.lm;
    }

    /**
     * Multiplies two moment_gugaussians when (a range of) the tail of the left
     * operand matches (a range of) the tail of the right operand.
     */
    template <typename TailIt1, typename TailIt2>
    friend void multiply_tails(const moment_gaussian_param& f,
                               const moment_gaussian_param& g,
                               index_range<TailIt1> f_tail,
                               index_range<TailIt2> g_tail,
                               moment_gaussian_param& r) {
      assert(f_tail.size() == g_tail.size());
      std::size_t n = f_tail.size();
      r.zero(f.head_size() + g.head_size(),
             f.tail_size() + g.tail_size() - n);
      span x(0, f.head_size());
      span y(f.head_size(), g.head_size());
      span v(0, f.tail_size());
      span w(f.tail_size(), g.tail_size() - n);
      subvec(r.mean, x) = f.mean;
      subvec(r.mean, y) = g.mean;
      submat(r.cov, x, x) = f.cov;
      submat(r.cov, y, y) = g.cov;
      submat(r.coef, x, v) = f.coef;
      submat(r.coef, y, f_tail) = subcols(g.coef, g_tail);
      submat(r.coef, y, w) = subcols(g.coef, complement(g_tail, g.tail_size()));
      r.lm = f.lm + g.lm;
    }

    /**
     * Computes a marginal or a maximum of the function represented by these
     * parameters over a subset of head dimensions, storing the result to an
     * output variable.
     *
     * \param marginal if true, will compute the marginal (false maximum)
     * \param head     the retained head dimensions
     * \param result   the output of the operation
     */
    template <typename It>
    void collapse(bool marginal,
                  index_range<It> head,
                  moment_gaussian_param& r) const {
      r.mean = subvec(mean, head);
      r.cov  = submat(cov, head, head);
      r.coef = subrows(coef, head);
      r.lm   = lm;
      if (!marginal) {
        r.lm += maximum() - r.maximum();
      }
    }

    /**
     * Computes a marginal or a maximum of the function represented by these
     * parameters over a subset of head dimensions, storing the result to an
     * output variable.
     *
     * \param marginal if true, will compute the marginal (false maximum)
     * \param head     the retained head dimensions
     * \param tail     the reshuffled tail dimensions
     * \param result   the output of the operation
     */
    template <typename It>
    void collapse(bool marginal,
                  index_range<It> head,
                  iref tail,
                  moment_gaussian_param& r) const {
      assert(tail.size() == tail_size());
      r.mean = subvec(mean, head);
      r.cov  = submat(cov, head, head);
      r.coef = submat(coef, head, tail);
      r.lm   = lm;
      if (!marginal) {
        r.lm += maximum() - r.maximum();
      }
    }

    /**
     * Computes the conditional of the marginal distribution represented by
     * these parameters, storing the result to an output variable.
     *
     * \param n      the dimensionality of the head
     * \param ws     the workspace for computing the conditional
     * \param result the output of the operation
     */
    void conditional(std::size_t n,
                     conditional_workspace& ws,
                     moment_gaussian_param& result) const {
      assert(is_marginal());
      assert(n <= head_size());
      std::size_t m = head_size() - n; // dimensionality of tail

      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      // where x are the head dimensions and y are the tail dimensions
      ws.sol_yx = cov.block(n, 0, m, n);
      ws.chol_yy.compute(cov.block(n, n, m, m));
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian::conditional: Cholesky decomposition failed"
        );
      }
      ws.chol_yy.solveInPlace(ws.sol_yx);

      // compute the parameters of the conditional
      result.mean = mean.segment(0, n);
      result.cov  = cov.block(0, 0, n, n);
      result.coef = ws.sol_yx.transpose();
      result.mean.noalias() -= ws.sol_yx.transpose() * mean.segment(n, m);
      result.cov.noalias()  -= ws.sol_yx.transpose() * cov.block(n, 0, m, n);
      result.lm = lm;
    }

    /**
     * Restricts the head of this conditional Gaussian to the given vector
     * and multiplies the result into an output variable.
     */
    template <typename It>
    void restrict_head_multiply(index_range<It> join_dims,
                                const dense_vector<RealType>& values,
                                moment_gaussian_param& result) const {
      assert(false);
    }

    /**
     * Restricts the moment_gaussian to a range of head dimensions and all
     * tail dimensions, storing the result to an output variable.
     *
     * \param x      the retained head dimensions
     * \param y      the restricted head dimensions
     * \param vals_y the values for the restricted head dimensions
     * \param vals_t the values for the tail dimensions
     * \param ws     the workspace for computing the conditional
     * \param result the output of the operation
     */
    template <typename RetainedIt, typename RestrictIt>
    void restrict_both(index_range<RetainedIt> x,
                       index_range<RestrictIt> y,
                       const dense_vector<RealType>& vals_y,
                       const dense_vector<RealType>& vals_t,
                       restrict_workspace& ws,
                       moment_gaussian_param& r) const {
      assert(x.size() + y.size() == head_size());
      assert(vals_y.size() == y.size());
      assert(vals_t.size() == tail_size());

      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      ws.cov_yy = submat(cov, y, y);
      ws.sol_yx = submat(cov, y, x);
      ws.chol_yy.compute(ws.cov_yy);
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "moment_gaussian restrict: Cholesky decomposition failed"
        );
      }
      if (!y.empty()) {
        ws.chol_yy.solveInPlace(ws.sol_yx);
      }

      // compute the residual over y (observation vec_y - the prediction)
      ws.res_y = vals_y - subvec(mean, y);
      if (!is_marginal()) {
        ws.res_y.noalias() -= subrows(coef, y) * vals_t;
      }

      // compute the output
      r.resize(x.size());
      r.mean.noalias() = subvec(mean, x) + ws.sol_yx.transpose() * ws.res_y;
      if (!is_marginal()) {
        r.mean.noalias() += subrows(coef, x) * vals_t;
      }
      r.cov.noalias() = submat(cov, x, x) - submat(cov, x, y) * ws.sol_yx;
      r.lm = lm -
        (y.size() * std::log(two_pi<RealType>()) + logdet(ws.chol_yy) +
         ws.res_y.dot(ws.chol_yy.solve(ws.res_y))) / RealType(2);
    }

    /**
     * Restricts a range of head dimensions of this momnet_gaussian to a vector,
     * storing the result to an output variable.
     *
     * \param x      the retained head dimensions
     * \param y      the restricted head dimensions
     * \param values the values for the restricted dimensions
     * \param r      the output
     */
    template <typename RetainedIt, typename RestrictIt>
    void restrict_head(index_range<RetainedIt> x,
                       index_range<RestrictIt> y,
                       const dense_vector<RealType>& values,
                       restrict_workspace& ws,
                       moment_gaussian_param& r) const {
      assert(is_marginal());
      restrict_both(x, y, values, dense_vector<RealType>(), ws, r);
    }

    /**
     * Restricts a range of tail dimensions of this moment_gaussian to a vector,
     * storing the result to an output variable.
     *
     * \param x the retained tail dimensions
     * \param y the restricted tail dimensions
     * \param values the values for the restricted tail dimensions
     * \param r the output of the operation
     */
    template <typename RetainedIt, typename RestrictIt>
    void restrict_tail(index_range<RetainedIt> x,
                       index_range<RestrictIt> y,
                       const dense_vector<RealType>& values,
                       moment_gaussian_param& r) const {
      assert(x.size() + y.size() == tail_size());
      r.mean.noalias() = mean + subcols(coef, y) * values;
      r.coef = subcols(coef, x);
      r.cov = cov;
      r.lm = lm;
    }

    /**
     * Returns the parameters reordered according to the given head
     * and tail indices.
     */
    template <typename It>
    void reorder(iref head, index_range<It> tail,
                 moment_gaussian_param& r) const {
      assert(head.size() == head_size());
      assert(tail.size() == tail_size());
      r.resize(head_size(), tail_size());
      r.mean = subvec(mean, head);
      r.cov  = submat(cov, head, head);
      r.coef = submat(coef, head, tail);
      r.lm   = lm;
    }

    /**
     * Prints the parameters to an output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const moment_gaussian_param& p) {
      out << p.mean << std::endl
          << p.cov << std::endl
          << p.coef << std::endl
          << p.lm;
      return out;
    }

  }; // class moment_gaussian_param

} // namespace libgm

#endif
