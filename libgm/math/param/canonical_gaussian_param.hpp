#ifndef LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP
#define LIBGM_CANONICAL_GAUSSIAN_PARAM_HPP

#include <libgm/math/constants.hpp>
#include <libgm/math/eigen/dynamic.hpp>
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
  template <typename T>
  struct canonical_gaussian_param {
    // Underlying representation
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    //! The information vector.
    vec_type eta;

    //! The information matrix.
    mat_type lambda;

    //! The log-multiplier.
    T lm;

    // Constructors and initialization
    //==========================================================================

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
      Eigen::LLT<mat_type>chol(mg.cov);
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

    // Accessors and function evaluation
    //==========================================================================

    //! Returns the size of the parameter struct.
    std::size_t size() const {
      assert(eta.rows() == lambda.rows() && eta.rows() == lambda.cols());
      return eta.rows();
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

    //! Reorders the parameter according to the given index.
    canonical_gaussian_param reorder(const matrix_index& map) const {
      assert(size() == map.size());
      canonical_gaussian_param result;
      set(result.eta, subvec(eta, map));
      set(result.lambda, submat(lambda, map, map));
      result.lm = lm;
      return result;
    }

    //! Returns the log-value for the given vector.
    T operator()(const vec_type& x) const {
      return -T(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
    }

    /**
     * Returns the normalization constant of the underlying factor assuming
     * that the parameters represent a marginal distribution.
     */
    T marginal() const {
      Eigen::LLT<mat_type> chol(lambda);
      return lm +
        (+ size() * std::log(two_pi<T>())
         - logdet(chol)
         + eta.dot(chol.solve(eta))) / T(2);
    }

    /**
     * Returns the maximum value attained by the underlying factor assuming
     * that the parameters represent a marginal distribution.
     */
    T maximum() const {
      vec_type vec;
      return maximum(vec);
    }

    /**
     * Returns the maximum value attained by the underlying factor and
     * the corresponding assignment, assuming that the parameters represent
     * a marginal distribution.
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
     * Returns the entropy for the distribution represented by this Gaussian.
     * The Gaussian must be normalizable.
     */
    T entropy() const {
      Eigen::LLT<mat_type> chol(lambda);
      return (size() * (std::log(two_pi<T>()) + T(1)) - logdet(chol)) / T(2);
    }

    /**
     * Returns the Kullback-Liebler divergnece from p to q.
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
     * Returns the max difference between the parameters eta and lambda
     * of two distributions.
     */
    friend T max_diff(const canonical_gaussian_param& f,
                      const canonical_gaussian_param& g) {
      T de = (f.eta - g.eta).array().abs().maxCoeff();
      T dl = (f.lambda - g.lambda).array().abs().maxCoeff();
      return std::max(de, dl);
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

    // Vector operations
    //==========================================================================

    //! Multiplies the parameters by a constant.
    canonical_gaussian_param& operator*=(T x) {
      eta *= x;
      lambda *= x;
      lm *= x;
      return *this;
    }

    //! Returns (1-a) * x + a * y.
    friend canonical_gaussian_param
    weighted_sum(const canonical_gaussian_param& x,
                 const canonical_gaussian_param& y, T a) {
      canonical_gaussian_param result;
      result.eta = (1-a) * x.eta + a * y.eta;
      result.lambda = (1-a) * x.lambda + a * y.lambda;
      result.lm = (1-a) * x.lm + a * y.lm;
      return result;
    }

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
    canonical_gaussian_join_inplace(matrix_index&& f_map)
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

    matrix_index f_map;
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
     */
    canonical_gaussian_join(matrix_index&& f_map,
                            matrix_index&& g_map,
                            std::size_t size)
      : f_op(std::move(f_map)),
        g_op(std::move(g_map)),
        size(size) { }

    //! Performs the join operation
    void operator()(const param_type& f, const param_type& g,
                    param_type& h) const {
      h.zero(size);
      f_op(h, f);
      g_op(h, g);
    }

    canonical_gaussian_join_inplace<T, libgm::plus_assign<> > f_op;
    canonical_gaussian_join_inplace<T, Update> g_op;
    std::size_t size;
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
    canonical_gaussian_maximum(matrix_index&& x, matrix_index&& y)
      : x(std::move(x)), y(std::move(y)) { }

    //! Performs the maximization operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) {
      // Compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      set(lam_yy, submat(f.lambda, y, y));
      set(sol_yx, submat(f.lambda, y, x));
      set(sol_y, subvec(f.eta, y));
      chol_yy.compute(lam_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error(
          "canonical_gaussian collapse: Cholesky decomposition failed"
        );
      }
      chol_yy.solveInPlace(sol_yx);
      chol_yy.solveInPlace(sol_y);

      // Compute the marginal parameters
      set(h.eta, subvec(f.eta, x));
      set(h.lambda, submat(f.lambda, x, x));
      if (y.contiguous()) {
        auto eta_y = subvec(f.eta, y).block(); // Eigen::Block
        h.eta.noalias() -= sol_yx.transpose() * eta_y;
        h.lm = f.lm + T(0.5) * sol_y.dot(eta_y);
      } else {
        auto eta_y = subvec(f.eta, y).plain(); // Eigen::Matrix
        h.eta.noalias() -= sol_yx.transpose() * eta_y;
        h.lm = f.lm + T(0.5) * sol_y.dot(eta_y);
      }
      submatrix<const dynamic_matrix<T>> lam_xy(f.lambda, x, y);
      if (lam_xy.contiguous()) {
        h.lambda.noalias() -= lam_xy.block() * sol_yx;
      } else {
        h.lambda.noalias() -= lam_xy.plain() * sol_yx;
      }
    }

    matrix_index x;
    matrix_index y;

    Eigen::LLT<dynamic_matrix<T>> chol_yy;
    dynamic_matrix<T> lam_yy;
    dynamic_matrix<T> sol_yx;
    dynamic_vector<T> sol_y;
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
    canonical_gaussian_marginal(matrix_index&& x, matrix_index&& y)
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
    typedef dynamic_vector<T> vec_type;
    typedef dynamic_matrix<T> mat_type;

    /**
     * Constructs a restrict operator.
     * \param x the retained indices in f
     * \param y the restricted indices in f
     * \param vec_y the assignment to the restricted arguments
     */
    canonical_gaussian_restrict(matrix_index&& x,
                                matrix_index&& y,
                                vec_type&& vec_y)
      : x(std::move(x)), y(std::move(y)) {
      vec_y.swap(this->vec_y);
    }

    //! Performs the restrict operation on f, storing the result in h.
    void operator()(const param_type& f, param_type& h) const {
      subvector<const vec_type> eta_y(f.eta, y);
      submatrix<const mat_type> lam_xy(f.lambda, x, y);
      submatrix<const mat_type> lam_yy(f.lambda, y, y);
      set(h.eta, subvec(f.eta, x));
      set(h.lambda, submat(f.lambda, x, x));
      h.lm = f.lm + eta_y.dot(vec_y);
      if (lam_xy.contiguous()) {
        h.eta.noalias() -= lam_xy.block() * vec_y;
      } else {
        h.eta.noalias() -= lam_xy.plain() * vec_y;
      }
      if (lam_yy.contiguous()) {
        h.lm -= 0.5 * vec_y.transpose() * lam_yy.block() * vec_y;
      } else {
        h.lm -= 0.5 * vec_y.transpose() * lam_yy.plain() * vec_y;
      }
    }

    matrix_index x;
    matrix_index y;
    dynamic_vector<T> vec_y;
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
    canonical_gaussian_restrict_join(matrix_index&& x,
                                     matrix_index&& y,
                                     matrix_index&& h_map,
                                     dynamic_vector<T>&& vec_y)
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
    matrix_index h_map;
    param_type tmp;
  };

} // namespace libgm

#endif
