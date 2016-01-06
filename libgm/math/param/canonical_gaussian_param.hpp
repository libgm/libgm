#ifndef LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP
#define LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>
#include <libgm/math/numerical_error.hpp>
#include <libgm/serialization/eigen.hpp>

#include <algorithm>

#include <Eigen/Cholesky>

namespace libgm {

  // Forward declaration
  template <typename T> class moment_gaussian_param;

  /**
   * The parameters of a Gaussian factor in the canonical parameterization.
   * This struct represents an unnormalized quadratic function
   * log f(x) = -0.5 x^T lambda x + eta^T x + c,
   * where lambda is a positive semidefinite matrix and c is a real number.
   *
   * \tparam T a real type for storing the coefficients
   */
  template <typename T = double>
  struct canonical_gaussian_param {
    // Public types
    //--------------------------------------------------------------------------
    //! The dense matrix type storing the information matrix.
    typedef real_matrix<T> mat_type;

    //! The dense vector type storing the information vector.
    typedef real_vector<T> vec_type;

    //! A vector of indices to a subset of parameters.
    typedef std::vector<std::size_t> index_type;

    //! The struct storing the temporaries for collapse computations.
    struct collapse_workspace {
      Eigen::LLT<mat_type> chol_yy;
      mat_type lam_yy;
      mat_type sol_yx;
      vec_type sol_y;
    };

    // The parameters
    //--------------------------------------------------------------------------

    //! The information vector.
    vec_type eta;

    //! The information matrix.
    mat_type lambda;

    //! The log-multiplier.
    T lm;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    //! Constructs an empty canonical Gaussian with given log-multiplier.
    canonical_gaussian_param(T lm = T(0))
      : lm(lm) { }

    //! Constructs uninitialized canonical Gaussian parameters with given size.
    explicit canonical_gaussian_param(std::size_t size)
      : lm(0) {
      resize(size);
    }

    //! Constructs canonical Gaussian parameters with the given log-multiplier.
    canonical_gaussian_param(std::size_t size, T lm)
      : lm(lm) {
      eta.setZero(size);
      lambda.setZero(size, size);
    }

    //! Constructor for the given natural parameters.
    canonical_gaussian_param(const vec_type& eta,
                             const mat_type& lambda,
                             T lm)
      : eta(eta), lambda(lambda), lm(lm) { }

    //! Copy constructor.
    canonical_gaussian_param(const canonical_gaussian_param& other) = default;

    //! Move constructor.
    canonical_gaussian_param(canonical_gaussian_param&& other) {
      swap(*this, other);
    }

    //! Conversion from a moment Gaussian.
    canonical_gaussian_param(const moment_gaussian_param<T>& cg) {
      *this = cg;
    }

    //! Assignment operator.
    canonical_gaussian_param& operator=(const canonical_gaussian_param& other) {
      if (this != &other) {
        eta = other.eta;
        lambda = other.lambda;
        lm = other.lm;
      }
      return *this;
    }

    //! Move assignment operator.
    canonical_gaussian_param& operator=(canonical_gaussian_param&& other) {
      swap(*this, other);
      return *this;
    }

    // Conversion from a moment Gaussian.
    canonical_gaussian_param& operator=(const moment_gaussian_param<T>& mg) {
      Eigen::LLT<mat_type> chol(mg.cov);
      if (chol.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian: Cannot invert the covariance matrix. "
          "Are you passing in a non-singular moment Gaussian distribution?"
        );
      }
      mat_type sol_xy = chol.solve(mg.coef);

      std::size_t m = mg.head_size();
      std::size_t n = mg.tail_size();
      resize(m + n);

      eta.segment(0, m) = chol.solve(mg.mean);
      eta.segment(m, n).noalias() = -sol_xy.transpose() * mg.mean;

      lambda.block(0, 0, m, m) = chol.solve(mat_type::Identity(m, m));
      lambda.block(0, m, m, n) = -sol_xy;
      lambda.block(m, 0, n, m) = -sol_xy.transpose();
      lambda.block(m, m, n, n).noalias() = mg.coef.transpose() * sol_xy;

      lm = mg.lm - T(0.5) * (eta.segment(0, m).dot(mg.mean)
                             + logdet(chol) + m * std::log(two_pi<T>()));
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
      lm = T(0);
    }

    //! Fills the pre-allocated parameters with 0.
    void zero() {
      eta.fill(T(0));
      lambda.fill(T(0));
      lm = T(0);
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
    T operator()(const vec_type& x) const {
      return -T(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
    }

    /**
     * Returns the normalization constant (in the log space) of a marginal
     * distribution represtend by this parameter struct.
     */
    T marginal() const {
      Eigen::LLT<mat_type> chol(lambda);
      return lm +
        (+ size() * std::log(two_pi<T>())
         - logdet(chol)
         + eta.dot(chol.solve(eta))) / T(2);
    }

    /**
     * Returns the maximum value (in the log space) attained by a marginal
     * distribution represtend by this parameter struct.
     */
    T maximum() const {
      vec_type vec;
      return maximum(vec);
    }

    /**
     * Returns the maximum value (in the log space) attained by a marginal
     * distribution represtend by this parameter struct and stores the
     * corresponding assignment to a vector.
     */
    T maximum(vec_type& vec) const {
      Eigen::LLT<mat_type> chol(lambda);
      if (chol.info() == Eigen::Success) {
        vec = chol.solve(eta);
        return lm + eta.dot(chol.solve(eta)) / T(2);
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
    T entropy() const {
      Eigen::LLT<mat_type> chol(lambda);
      return (size() * (std::log(two_pi<T>()) + T(1)) - logdet(chol)) / T(2);
    }

    /**
     * Returns the Kullback-Leibler divergnece from p to q.
     */
    friend T kl_divergence(const canonical_gaussian_param& p,
                           const canonical_gaussian_param& q) {
      assert(p.size() == q.size());
      Eigen::LLT<mat_type> cholp(p.lambda);
      Eigen::LLT<mat_type> cholq(q.lambda);
      vec_type mup = cholp.solve(p.eta);
      vec_type muq = cholq.solve(q.eta);
      auto identity = mat_type::Identity(p.size(), p.size());
      T trace = (q.lambda.array() * cholp.solve(identity).array()).sum();
      T means = (mup - muq).transpose() * q.lambda * (mup - muq);
      T logdets = logdet(cholp) - logdet(cholq);
      return (trace + means + logdets - p.size()) / T(2);
    }

    /**
     * Returns the maximum difference between the parameters eta and lambda
     * of two factors.
     */
    friend T max_diff(const canonical_gaussian_param& f,
                      const canonical_gaussian_param& g) {
      T deta = (f.eta - g.eta).array().abs().maxCoeff();
      T dlam = (f.lambda - g.lambda).array().abs().maxCoeff();
      return std::max(deta, dlam);
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Multiplies each parameter by a constant in-place.
     */
    canonical_gaussian_param& operator*=(T x) {
      eta *= x;
      lambda *= x;
      lm *= x;
      return *this;
    }

    /**
     * Updates the result with the parameters in this struct.
     */
    template <typename UpdateOp>
    void update(UpdateOp update_op,
                const index_type& idx,
                canonical_gaussian_param& result) const {
      update_op(subvec(result.eta, idx), eta);
      update_op(submat(result.lambda, idx, idx), lambda);
      update_op(result.lm, lm);
    }

    /**
     * Transforms the parameters by a unary function in-place.
     */
    template <typename VectorOp, typename ScalarOp>
    void transform(VectorOp vector_op, ScalarOp scalar_op) {
      vector_op.update(eta);
      vector_op.update(lambda);
      scalar_op.update(lm);
    }

    /**
     * Transforms the parameters by a unary function and stores the result
     * to the output parameter struct.
     */
    template <typename VectorOp, typename ScalarOp>
    void transform(VectorOp vector_op,
                   ScalarOp scalar_op,
                   canonical_gaussian_param& result) const {
      result.eta    = vector_op(eta);
      result.lambda = vector_op(lambda);
      result.lm     = scalar_op(lm);
    }

    /**
     * Transforms the parameters by a unary function and updates the result
     * at the specified indices.
     */
    template <typename VectorOp, typename ScalarOp, typename UpdateOp>
    void transform_update(VectorOp vector_op,
                          ScalarOp scalar_op,
                          UpdateOp update_op,
                          const index_type& idx,
                          canonical_gaussian_param& result) const {
      update_op(subvec(result.eta, idx), vector_op(eta).eval());
      update_op(submat(result.lambda, idx, idx), vector_op(lambda).eval());
      update_op(result.lm, scalar_op(lm));
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
    void collapse(const index_type& x,
                  const index_type& y,
                  bool marginal,
                  collapse_workspace& ws,
                  canonical_gaussian_param& result) const {
      assert(size() == x.size() + y.size());

      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      submat(lambda, y, y).eval_to(ws.lam_yy);
      submat(lambda, y, x).eval_to(ws.sol_yx);
      subvec(eta, y).eval_to(ws.sol_y);
      ws.chol_yy.compute(ws.lam_yy);
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      ws.chol_yy.solveInPlace(ws.sol_yx);
      ws.chol_yy.solveInPlace(ws.sol_y);

      // store the aggregate parameters
      subvec(eta, x).eval_to(result.eta);
      submat(lambda, x, x).eval_to(result.lambda);
      result.eta.noalias() -= ws.sol_yx.transpose() * subvec(eta, y).ref();
      result.lambda.noalias() -= submat(lambda, x, y).ref() * ws.sol_yx;
      result.lm = lm + T(0.5) * subvec(eta, y).dot(ws.sol_y);

      // adjust the log-likelihood if computing the marginal
      if (marginal) {
        result.lm += T(0.5) * std::log(two_pi<T>()) * y.size();
        result.lm -= T(0.5) * logdet(ws.chol_yy);
      }
    }

    /**
     * Computes the conditional of the marginal distribution represented by
     * these parameters, storing the result to an output variable.
     *
     * \param idx    the indices of the (head, tail) dimensions in the input
     * \param n      the dimensionality of tail
     * \param ws     the workspace for computing the aggregate
     * \param result the output of the operation
     */
    void conditional(const index_type& idx,
                     std::size_t n,
                     collapse_workspace& ws,
                     canonical_gaussian_param& result) const {
      assert(size() == idx.size() && n <= idx.size());
      std::size_t m = idx.size() - n; // dimensionality of head
      subvec(eta, idx).eval_to(result.eta);
      submat(lambda, idx, idx).eval_to(result.lambda);

      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition, where y are the head dimensions, and
      // x are the tail dimensions
      ws.lam_yy = result.lambda.block(0, 0, m, m);
      ws.sol_yx = result.lambda.block(0, m, m, n);
      ws.sol_y = result.eta.segment(0, m);
      ws.chol_yy.compute(ws.lam_yy);
      if (ws.chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      ws.chol_yy.solveInPlace(ws.sol_yx);
      ws.chol_yy.solveInPlace(ws.sol_y);

      // compute the tail parameters
      result.eta.segment(m, n).noalias()
        = ws.sol_yx.transpose() * result.eta.segment(0, m);
      result.lambda.block(m, m, n, n).noalias()
        = result.lambda.block(m, 0, n, m) * ws.sol_yx;
      result.lm
        = T(0.5) * (logdet(ws.chol_yy) - std::log(two_pi<T>()) * m
                    - result.eta.segment(0, m).dot(ws.sol_y));
    }

    /**
     * Computes the restriction of the function represented by these parameters
     * to a subset of dimensions.
     *
     * \param x      the indices of the retained dimensions
     * \param y      the indices of the eliminated dimensions
     * \param vec_y  the assignment to the eliminated dimensions
     * \param result the output of the operation
     */
    void restrict(const index_type& x,
                  const index_type& y,
                  const vec_type& vec_y,
                  canonical_gaussian_param& result) const {
      subvec(eta, x).eval_to(result.eta);
      submat(lambda, x, x).eval_to(result.lambda);
      result.eta.noalias() -= submat(lambda, x, y).ref() * vec_y;
      result.lm = lm + subvec(eta, y).dot(vec_y)
        - 0.5 * vec_y.transpose() * submat(lambda, y, y).ref() * vec_y;
    }

    /**
     * Computes the restriction of the function represented by these parameters
     * to a subset of dimensions, updating the output with the result.
     *
     * \param x         the indices of the retained dimensions
     * \param y         the indices of the eliminated dimensions
     * \param vec       the assignment to the eliminated dimensions
     * \param update_op the update operation such as plus_assign<>
     * \param idx       the indices of the retained dimensions in the result
     * \param result    the output of the operation
     */
    template <typename UpdateOp>
    void restrict_update(const index_type& x,
                         const index_type& y,
                         const vec_type& vec,
                         UpdateOp update_op,
                         const index_type& idx,
                         canonical_gaussian_param& result) const {
      assert(idx.size() == x.size());
      update_op(subvec(result.eta, idx), subvec(eta, x));
      update_op(subvec(result.eta, idx), vec_type(-submat(lambda, x, y).ref() * vec));
      update_op(submat(result.lambda, idx, idx), submat(lambda, x, x));
      result.lm += lm + subvec(eta, y).dot(vec)
        - 0.5 * vec.transpose() * submat(lambda, y, y).ref() * vec;
    }

    /**
     * Returns the parameter reordered according to the given index.
     */
    canonical_gaussian_param reorder(const std::vector<std::size_t>& map) const {
      assert(size() == map.size());
      canonical_gaussian_param result;
      subvec(eta, map).eval_to(result.eta);
      submat(lambda, map, map).eval_to(result.lambda);
      result.lm = lm;
      return result;
    }

    /**
     * Returns (1-a) * x + a * y.
     */
    friend canonical_gaussian_param
    weighted_sum(const canonical_gaussian_param& x,
                 const canonical_gaussian_param& y, T a) {
      canonical_gaussian_param result;
      result.eta = (1-a) * x.eta + a * y.eta;
      result.lambda = (1-a) * x.lambda + a * y.lambda;
      result.lm = (1-a) * x.lm + a * y.lm;
      return result;
    }

#if 0
    /**
     * Ensures that the information matrix is PSD, i.e., this factor represents
     * a valid likelihood.
     * \param mean the mean of the joint distribution
     * @return  True if the information matrix was already PSD and false if
     *          it was not and was adjusted.
     */
    bool enforce_psd(const vec& mean);
#endif

  }; // struct canonical_gaussian_param

  /**
   * Prints the parameters to an output stream.
   */
  template <typename T>
  std::ostream&
  operator<<(std::ostream& out, const canonical_gaussian_param<T>& p) {
    out << p.eta << std::endl
        << p.lambda << std::endl
        << p.lm;
    return out;
  }

  // Join operations
  //============================================================================

  /**
   * A class that joins one canonical Gaussian with another canonical
   * Gaussian or constant in-place.
   */
  template <typename T, typename Update>
  struct canonical_gaussian_join_inplace {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs the operation for joining with a constant.
     */
    canonical_gaussian_join_inplace() { }

    /**
     * Constructs the operation for joining with a factor.
     * \param f_map mapping of f's indices to those of the result.
     */
    canonical_gaussian_join_inplace(std::vector<std::size_t>&& f_map)
      : f_map(std::move(f_map)) { }

    //! Joins the result in-place with a Gaussian.
    void operator()(param_type& result, const param_type& f) const {
      Update update;
      update(subvec(result.eta, f_map), f.eta);
      update(submat(result.lambda, f_map, f_map), f.lambda);
      update(result.lm, f.lm);
    }

    //! Joins the result in-place with a constant in log-space.
    void operator()(param_type& result, T x) const {
      Update update;
      update(result.lm, x);
    }

    //! Joins the constant in log-space in-place with the result.
    void operator()(T x, param_type& result) const {
      Update update;
      update(result.eta);
      update(result.lambda);
      update(result.lm);
      result.lm += x;
    }

    std::vector<std::size_t> f_map;
  };

  /**
   * A class that joins the parameters of two canonical Gaussians.
   */
  template <typename T, typename Update>
  struct canonical_gaussian_join {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs the operation.
     * \param f_map mapping of f's indices to those of the result
     * \param g_map mapping of g's indices to those of the result
     * \param ndims the number of dimensions of the result
     */
    canonical_gaussian_join(std::vector<std::size_t>&& f_map,
                            std::vector<std::size_t>&& g_map,
                            std::size_t ndims)
      : f_op(std::move(f_map)),
        g_op(std::move(g_map)),
        ndims(ndims) { }

    //! Performs the join operation
    void operator()(const param_type& f, const param_type& g,
                    param_type& h) const {
      h.zero(ndims);
      f_op(h, f);
      g_op(h, g);
    }

    canonical_gaussian_join_inplace<T, libgm::plus_assign<> > f_op;
    canonical_gaussian_join_inplace<T, Update> g_op;
    std::size_t ndims;
  };

  // Aggregate operations
  //============================================================================

  /**
   * A class that computes the maximum of a canonical Gaussian.
   */
  template <typename T>
  struct canonical_gaussian_maximum {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs a maximization operator.
     * \param x the retained indices in f
     * \param y the eliminated indices in f
     */
    canonical_gaussian_maximum(std::vector<std::size_t>&& x,
                               std::vector<std::size_t>&& y)
      : x(std::move(x)), y(std::move(y)) { }

    //! Performs the maximization operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      // Compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      submat(f.lambda, y, y).eval_to(lam_yy);
      submat(f.lambda, y, x).eval_to(sol_yx);
      subvec(f.eta, y).eval_to(sol_y);
      chol_yy.compute(lam_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);
      chol_yy.solveInPlace(sol_y);

      // Compute the marginal parameters
      subvec(f.eta, x).eval_to(h.eta);
      h.eta.noalias() -= sol_yx.transpose() * subvec(f.eta, y).ref();
      submat(f.lambda, x, x).eval_to(h.lambda);
      h.lambda.noalias() -= submat(f.lambda, x, y).ref() * sol_yx;
      h.lm = f.lm + T(0.5) * subvec(f.eta, y).dot(sol_y);
    }

    std::vector<std::size_t> x;
    std::vector<std::size_t> y;

    Eigen::LLT<real_matrix<T>> chol_yy;
    real_matrix<T> lam_yy;
    real_matrix<T> sol_yx;
    real_vector<T> sol_y;
  };

  /**
   * A class that computes the marginal of a canonical Gaussian.
   */
  template <typename T>
  struct canonical_gaussian_marginal {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs a marginalization operator.
     * \param x the retained indices in f
     * \param y the eliminated indices in f
     */
    canonical_gaussian_marginal(std::vector<std::size_t>&& x,
                                std::vector<std::size_t>&& y)
      : maximum_op(std::move(x), std::move(y)) { }

    //! Performs the marginalization operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      maximum_op(f, h);
      h.lm += T(0.5) * (f.size() - h.size()) * std::log(two_pi<T>());
      h.lm -= T(0.5) * logdet(maximum_op.chol_yy);
   }

    canonical_gaussian_maximum<T> maximum_op;
  };

  // Restrict operations
  //============================================================================

  /**
   * A class that restricts a canonical Gaussian.
   */
  template <typename T>
  struct canonical_gaussian_restrict {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs a restrict operator.
     * \param x the retained indices in f
     * \param y the restricted indices in f
     * \param vec_y the assignment to the restricted arguments
     */
    canonical_gaussian_restrict(std::vector<std::size_t>&& x,
                                std::vector<std::size_t>&& y,
                                real_vector<T>&& vec_y)
      : x(std::move(x)), y(std::move(y)) {
      vec_y.swap(this->vec_y);
    }

    //! Performs the restrict operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) const {
      subvec(f.eta, x).eval_to(h.eta);
      submat(f.lambda, x, x).eval_to(h.lambda);
      h.eta.noalias() -= submat(f.lambda, x, y).ref() * vec_y;
      h.lm = f.lm + subvec(f.eta, y).dot(vec_y)
        - 0.5 * vec_y.transpose() * submat(f.lambda, y, y).ref() * vec_y;
    }

    std::vector<std::size_t> x;
    std::vector<std::size_t> y;
    real_vector<T> vec_y;
  };

  /**
   * A class that restricts a canonical Gaussian and joins the result with
   * another Gaussian.
   */
  template <typename T, typename Update>
  struct canonical_gaussian_restrict_join {
    typedef canonical_gaussian_param<T> param_type;

    /**
     * Constructs a restrict operator.
     * \param x the retained indices in f
     * \param y restricted indices in f
     * \param h_map the mapping from the retained indices to the result
     * \param vec_y the vector of restricted values
     */
    canonical_gaussian_restrict_join(std::vector<std::size_t>&& x,
                                     std::vector<std::size_t>&& y,
                                     std::vector<std::size_t>&& h_map,
                                     real_vector<T>&& vec_y)
      : restrict_op(std::move(x), std::move(y), std::move(vec_y)),
        h_map(std::move(h_map)) { }

    //! Restricts f and joins the result with h.
    void operator()(const param_type& f, param_type& h) {
      Update update;
      restrict_op(f, tmp);
      update(subvec(h.eta, h_map), tmp.eta);
      update(submat(h.lambda, h_map, h_map), tmp.lambda);
      update(h.lm, tmp.lm);
    }

    canonical_gaussian_restrict<T> restrict_op;
    std::vector<std::size_t> h_map;
    param_type tmp;
  };

} // namespace libgm

#endif
