#ifndef LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP
#define LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/range/index_range.hpp>
#include <libgm/serialization/eigen.hpp>

#include <algorithm>

#include <Eigen/Cholesky>

namespace libgm {

  // Forward declaration
  template <typename RealType> struct moment_gaussian_param;

  /**
   * The parameters of a Gaussian factor in the canonical parameterization.
   * This struct represents an unnormalized quadratic function
   * log f(x) = -0.5 x^T lambda x + eta^T x + c,
   * where lambda is a positive semidefinite matrix and c is a real number.
   *
   * \tparam RealType a real type for storing the coefficients
   */
  template <typename RealType = double>
  struct canonical_gaussian_param {
    // Public types
    //--------------------------------------------------------------------------

    //! The type storing the parameters.
    typedef RealType value_type;

    //! The type of the LLT Cholesky decomposition object.
    typedef Eigen::LLT<dense_matrix<RealType> > cholesky_type;

    //! The struct storing the temporaries for collapse computations.
    struct collapse_workspace {
      cholesky_type chol_yy;
      dense_matrix<RealType> lam_yy;
      dense_matrix<RealType> sol_yx;
      dense_vector<RealType> sol_y;
    };

    // The parameters
    //--------------------------------------------------------------------------

    //! The information vector.
    dense_vector<RealType> eta;

    //! The information matrix.
    dense_matrix<RealType> lambda;

    //! The log-multiplier.
    RealType lm;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    //! Constructs an empty canonical Gaussian with given log-multiplier.
    explicit canonical_gaussian_param(RealType lm = RealType(0))
      : lm(lm) { }

    //! Constructs uninitialized canonical Gaussian parameters with given size.
    explicit canonical_gaussian_param(std::size_t size)
      : lm(0) {
      resize(size);
    }

    //! Constructs canonical Gaussian parameters with the given log-multiplier.
    canonical_gaussian_param(std::size_t size, RealType lm)
      : lm(lm) {
      eta.setZero(size);
      lambda.setZero(size, size);
    }

    //! Constructor for the given natural parameters.
    canonical_gaussian_param(const dense_vector<RealType>& eta,
                             const dense_matrix<RealType>& lambda,
                             RealType lm)
      : eta(eta), lambda(lambda), lm(lm) {
      check();
    }

    //! Conversion from a moment Gaussian.
    canonical_gaussian_param(const moment_gaussian_param<RealType>& cg) {
      *this = cg;
    }

    //! Conversion from a moment Gaussian.
    canonical_gaussian_param&
    operator=(const moment_gaussian_param<RealType>& mg) {
      cholesky_type chol(mg.cov);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian: Cannot invert the covariance matrix. "
          "Are you passing in a non-singular moment Gaussian distribution?"
        );
      }
      dense_matrix<RealType> sol_xy = chol.solve(mg.coef);

      std::size_t m = mg.head_size();
      std::size_t n = mg.tail_size();
      resize(m + n);

      eta.segment(0, m) = chol.solve(mg.mean);
      eta.segment(m, n).noalias() = -sol_xy.transpose() * mg.mean;

      lambda.block(0, 0, m, m) = chol.solve(dense_matrix<RealType>::Identity(m, m));
      lambda.block(0, m, m, n) = -sol_xy;
      lambda.block(m, 0, n, m) = -sol_xy.transpose();
      lambda.block(m, m, n, n).noalias() = mg.coef.transpose() * sol_xy;

      lm = mg.lm - (m * std::log(two_pi<RealType>()) + logdet(chol)
                    + eta.segment(0, m).dot(mg.mean)) / RealType(2);
      return *this;
    }

    //! Swaps the two sets of parameters.
    friend void swap(canonical_gaussian_param& a, canonical_gaussian_param& b) {
      a.eta.swap(b.eta);
      a.lambda.swap(b.lambda);
      std::swap(a.lm, b.lm);
    }

    //! Serializes the parameters to an archive.
    void save(oarchive& ar) const {
      ar << eta << lambda << lm;
    }

    //! Deserializes the parameters from an archive.
    void load(iarchive& ar) {
      ar >> eta >> lambda >> lm;
    }

    //! Resizes the parameter vector to the given vector size.
    void resize(std::size_t n) {
      eta.resize(n);
      lambda.resize(n, n);
    }

    //! Initializes the parameter struct to 0 with given vector size.
    void zero(std::size_t n) {
      eta.setZero(n);
      lambda.setZero(n, n);
      lm = RealType(0);
    }

    //! Fills the pre-allocated parameters with 0.
    void zero() {
      eta.fill(RealType(0));
      lambda.fill(RealType(0));
      lm = RealType(0);
    }

    //! Throws an exception if the information vector and matrix do not match.
    void check() const {
      if (lambda.rows() != lambda.cols()) {
        throw std::invalid_argument(
          "The information matrix is not square."
        );
      }
      if (eta.rows() != lambda.rows()) {
        throw std::invalid_argument(
          "The information vector and matrix have incompatible sizes."
        );
      }
    }

    //! Throws an exception if the vector and matrix dimensions are invalid.
    void check_size(std::size_t n) const {
      if (lambda.rows() != lambda.cols()) {
        throw std::logic_error("The information matrix is not square.");
      }
      if (lambda.rows() != n) {
        throw std::logic_error("Invalid size of the information matrix.");
      }
      if (eta.rows() != n) {
        throw std::logic_error("Invalid size of the information vector.");
      }
    }

    //! Returns true if the two parameter structs are identical.
    friend bool operator==(const canonical_gaussian_param& f,
                           const canonical_gaussian_param& g) {
      return f.eta == g.eta && f.lambda == g.lambda && f.lm == g.lm;
    }

    //! Returns true if the two parameter structs are not identical.
    friend bool operator!=(const canonical_gaussian_param& f,
                           const canonical_gaussian_param& g) {
      return !(f == g);
    }

    // Accessors and function evaluation
    //--------------------------------------------------------------------------

    //! Returns the dimensionality of the Gaussian.
    std::size_t size() const {
      assert(eta.rows() == lambda.rows() && eta.rows() == lambda.cols());
      return eta.rows();
    }

    //! Returns the log-value for the given vector.
    RealType operator()(const dense_vector<RealType>& x) const {
      return -RealType(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
    }

    /**
     * Returns the normalization constant (in the log space) of a marginal
     * distribution represtend by this parameter struct.
     */
    RealType marginal() const {
      cholesky_type chol(lambda);
      return lm +
        (+ size() * std::log(two_pi<RealType>())
         - logdet(chol)
         + eta.dot(chol.solve(eta))) / RealType(2);
    }

    /**
     * Returns the maximum value (in the log space) attained by a marginal
     * distribution represtend by this parameter struct.
     */
    RealType maximum() const {
      dense_vector<RealType> vec;
      return maximum(vec);
    }

    /**
     * Returns the maximum value (in the log space) attained by a marginal
     * distribution represtend by this parameter struct and stores the
     * corresponding assignment to a vector.
     */
    RealType maximum(dense_vector<RealType>& vec) const {
      cholesky_type chol(lambda);
      if (chol.info() == Eigen::Success) {
        vec = chol.solve(eta);
        return lm + eta.dot(chol.solve(eta)) / RealType(2);
      } else {
        throw numerical_error(
          "canonical_gaussian::maximum(): Cholesky decomposition failed"
        );
      }
    }

    /**
     * Returns the entropy for the marginal distribution represented by these
     * parameters.
     */
    RealType entropy() const {
      cholesky_type chol(lambda);
      return (size() * (std::log(two_pi<RealType>()) + RealType(1))
              - logdet(chol)) / RealType(2);
    }

    /**
     * Returns the Kullback-Leibler divergnece from p to q.
     */
    friend RealType kl_divergence(const canonical_gaussian_param& p,
                                  const canonical_gaussian_param& q) {
      assert(p.size() == q.size());
      cholesky_type cholp(p.lambda);
      cholesky_type cholq(q.lambda);
      dense_vector<RealType> mup = cholp.solve(p.eta);
      dense_vector<RealType> muq = cholq.solve(q.eta);
      auto identity = dense_matrix<RealType>::Identity(p.size(), p.size());
      RealType trace = (q.lambda.array() * cholp.solve(identity).array()).sum();
      RealType means = (mup - muq).transpose() * q.lambda * (mup - muq);
      RealType logdets = logdet(cholp) - logdet(cholq);
      return (trace + means + logdets - p.size()) / RealType(2);
    }

    /**
     * Returns the maximum difference between the parameters eta and lambda
     * of two factors.
     */
    friend RealType max_diff(const canonical_gaussian_param& f,
                             const canonical_gaussian_param& g) {
      RealType deta = (f.eta - g.eta).array().abs().maxCoeff();
      RealType dlam = (f.lambda - g.lambda).array().abs().maxCoeff();
      return std::max(deta, dlam);
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Multiplies each parameter by a constant in-place.
     */
    canonical_gaussian_param& operator*=(RealType x) {
      eta *= x;
      lambda *= x;
      lm *= x;
      return *this;
    }

    /**
     * Sets the information vector, matrix, and log-multiplier to the specified
     * vector expression, matrix expression, and constant.
     */
    template <typename Vector, typename Matrix>
    void assign(const Eigen::MatrixBase<Vector>& ivec,
                const Eigen::MatrixBase<Matrix>& imat,
                RealType logmult) {
      eta = ivec;
      lambda = imat;
      lm = logmult;
    }

    /**
     * Updates the information vector, matrix, and log-multiplier using the
     * specified mutating operation and parameter expressions.
     */
    template <typename UpdateOp, typename Vector, typename Matrix>
    void update(UpdateOp update_op,
                const Eigen::MatrixBase<Vector>& ivec,
                const Eigen::MatrixBase<Matrix>& imat,
                RealType logmult) {
      update_op(eta, ivec);
      update_op(lambda, imat);
      update_op(lm, logmult);
    }

    /**
     * Updates a range of dimensions of the information vector, matrix, and
     * log-multiplier using a mutating operation and parameter expressions.
     */
    template<typename UpdateOp, typename It, typename Vector, typename Matrix>
    void update(UpdateOp update_op,
                index_range<It> dims,
                const Eigen::MatrixBase<Vector>& ivec,
                const Eigen::MatrixBase<Matrix>& imat,
                RealType logmult) {
      update_op(subvec(eta, dims), ivec);
      update_op(submat(lambda, dims, dims), imat);
      update_op(lm, logmult);
    }

    /**
     * Computes an integral or a maximum of the function represented by these
     * parameters over a subset of dimensions, storing the result to an output
     * variable.
     *
     * \param x        the indices of the retained dimensions
     * \param y        the indices of the eliminated dimensions
     * \param marginal if true, will compute the marginal (false maximum)
     * \param ws       the workspace for storing the temporaries
     * \param result   the output of the operation
     */
    template <typename RetainIt, typename EliminIt>
    void collapse(bool marginal,
                  index_range<RetainIt> x,
                  index_range<EliminIt> y,
                  collapse_workspace& ws,
                  canonical_gaussian_param& result) const {
      assert(x.size() + y.size() == size());

      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      ws.lam_yy = submat(lambda, y, y);
      ws.sol_yx = submat(lambda, y, x);
      ws.sol_y = subvec(eta, y);
      ws.chol_yy.compute(ws.lam_yy);
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      if (!y.empty()) {
        ws.chol_yy.solveInPlace(ws.sol_yx);
        ws.chol_yy.solveInPlace(ws.sol_y);
      }

      // store the aggregate parameters
      result.eta = subvec(eta, x);
      result.lambda = submat(lambda, x, x);
      result.eta.noalias() -= ws.sol_yx.transpose() * subvec(eta, y);
      result.lambda.noalias() -= submat(lambda, x, y) * ws.sol_yx;
      result.lm = lm + RealType(0.5) * subvec(eta, y).dot(ws.sol_y);

      // adjust the log-likelihood if computing the marginal
      if (marginal) {
        result.lm += RealType(0.5) * std::log(two_pi<RealType>()) * y.size();
        result.lm -= RealType(0.5) * logdet(ws.chol_yy);
      }
    }

    /**
     * Computes the conditional of the marginal distribution represented by
     * these parameters, storing the result to an output variable.
     *
     * \param n      the dimensionality of the head
     * \param ws     the workspace for computing the aggregate
     * \param result the output of the operation
     */
    void conditional(std::size_t n,
                     collapse_workspace& ws,
                     canonical_gaussian_param& result) const {
      assert(n <= size());
      std::size_t m = size() - n; // dimensionality of tail

      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition, where y are the head dimensions, and
      // x are the tail dimensions
      ws.lam_yy = lambda.block(0, 0, n, n);
      ws.sol_yx = lambda.block(0, n, n, m);
      ws.sol_y = eta.segment(0, n);
      ws.chol_yy.compute(ws.lam_yy);
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      ws.chol_yy.solveInPlace(ws.sol_yx);
      ws.chol_yy.solveInPlace(ws.sol_y);

      // compute the tail parameters
      if (&result != this) {
        result.eta = eta;
        result.lambda = lambda;
      }
      result.eta.segment(n, m).noalias()
        = ws.sol_yx.transpose() * eta.segment(0, n);
      result.lambda.block(n, n, m, m).noalias()
        = ws.sol_yx.transpose() * lambda.block(0, n, n, m);
      result.lm
        = RealType(0.5) * (logdet(ws.chol_yy) - std::log(two_pi<RealType>()) * n
                           - eta.segment(0, n).dot(ws.sol_y));
    }

    /**
     * Computes the restriction of the function represented by these parameters
     * to a subset of dimensions.
     *
     * \param x      the retained dimensions
     * \param y      the restricted dimensions
     * \param values the assignment to the restricted dimensions
     * \param r      the output of the operation
     */
    template <typename RetainedIt, typename RestrictIt>
    void restrict(index_range<RetainedIt> x,
                  index_range<RestrictIt> y,
                  const dense_vector<RealType>& values,
                  canonical_gaussian_param& r) const {
      assert(x.size() + y.size() == size());
      r.eta.noalias() = subvec(eta, x) - submat(lambda, x, y) * values;
      r.lambda = submat(lambda, x, x);
      r.lm = lm + subvec(eta, y).dot(values)
        - RealType(0.5) * values.transpose() * submat(lambda, y, y) * values;
    }

    /**
     * Returns the parameter reordered according to the given index.
     */
    void reorder(iref dims, canonical_gaussian_param& r) const {
      assert(dims.size() == size());
      r.eta = subvec(eta, dims);
      r.lambda = submat(lambda, dims, dims);
      r.lm = lm;
    }

    /**
     * Returns (1-a) * x + a * y.
     */
    friend canonical_gaussian_param
    weighted_sum(const canonical_gaussian_param& x,
                 const canonical_gaussian_param& y, RealType a) {
      canonical_gaussian_param result;
      result.eta = (1-a) * x.eta + a * y.eta;
      result.lambda = (1-a) * x.lambda + a * y.lambda;
      result.lm = (1-a) * x.lm + a * y.lm;
      return result;
    }

    /**
     * Prints the parameters to an output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const canonical_gaussian_param& p) {
      out << p.eta << std::endl
          << p.lambda << std::endl
          << p.lm;
      return out;
    }

  }; // struct canonical_gaussian_param

} // namespace libgm

#endif
