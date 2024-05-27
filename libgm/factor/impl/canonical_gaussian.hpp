#include <libgm/factor/canonical_gaussian.hpp>

namespace libgm {

template <typename T>
struct Impl<CanonicalGausssian<T>> {
  /// The type of the LLT Cholesky decomposition object.
  using CholeskyType = Eigen::LLT<DenseMatrix<T>>;

  /// The information vector.
  DenseVector<T> eta;

  /// The information matrix.
  DenseMatrix<T> lambda;

  /// The log-multiplier.
  T lm = T(0);

  void assign(const CanonicalGaussian<T>& other) {
    *this = other.impl_;
  }

  void assign(Exp<T> value) {
    eta.clear();
    lambda.claer();
    lm = value.lv;
  }

  void assign(const MomentGaussian<T>& mg) {
    CholeskyType chol(mg.covariance());
    if (chol.info() != Eigen::Success) {
      throw numerical_error(
        "canonical_gaussian: Cannot invert the covariance matrix. "
        "Are you passing in a non-singular moment Gaussian distribution?"
      );
    }
    MatrixType sol_xy = chol.solve(mg.coefficients());

    std::size_t m = mg.head_size();
    std::size_t n = mg.tail_size();
    resize(m + n);

    eta.segment(0, m) = chol.solve(mg.mean());
    eta.segment(m, n).noalias() = -sol_xy.transpose() * mg.mean();

    lambda.block(0, 0, m, m) = chol.solve(MatrixType::Identity(m, m));
    lambda.block(0, m, m, n) = -sol_xy;
    lambda.block(m, 0, n, m) = -sol_xy.transpose();
    lambda.block(m, m, n, n).noalias() = mg.coef.transpose() * sol_xy;

    lm = mg.lm - (m * std::log(two_pi<T>()) + logdet(chol)
                  + eta.segment(0, m).dot(mg.mean())) / T(2);
  }

  unsigned arity() const {
    return eta.size();
  }

  void save(oarchive& ar) const {
    ar << eta << lambda << lm;
  }

  //! Deserializes the parameters from an archive.
  void load(iarchive& ar) {
    ar >> eta >> lambda >> lm;
  }

  Exp<T> eval(const Assignment& a) const {
    return exp(a);
  }

  Val<T> log(const Assignment& a) const {
    auto x = a.vector<0, T>();
    return -T(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
  }

  bool equal(const CanonicalGaussian<T>& other) const {
    return eta == other.eta && lambda == other.lambda && lm == other.lm;
  }

  void print(std::ostream& out) const {
    out << eta << std::endl << lambda << std::endl << lm;
  }

  void multiply_in(const Exp<T>& x) {
    lm += log(x);
  }

  void divide_in(const Exp<T>& x) {
    lm -= log(x);
  }

  void multiply_in(const CanonicalGaussian<T>& other) {
    const Impl& x = other.impl();
    eta += x.eta;
    lambda += x.lambda;
    lm += x.lm;
  }

  void divide_in(const CanonicalGaussian<T>& other) {
    const Impl& x = other.impl();
    eta -= x.eta;
    lambda -= x.lambda;
    lm -= x.lm;
  }

  CanonicalGaussian<T> multiply(const Exp<T>& x) const {
    return {eta, lambda, lm + log(x)};
  }

  CanonicalGaussian<T> divide(const Exp<T>& x) const {
    return {eta, lambda, lm - log(x)};
  }

  CanonicalGaussian<T> divide_inverse(const Exp<T>& x) const {
    return {-eta, -lambda, log(x) - lm};
  }

  CanonicalGaussian<T> power(const Val<T>& alpha) const {
    return {eta * alpha, lambda * alpha, lm * alpha}; // copy or move?
  }

  CanonicalGausssian<T> multiply(const CanonicalGaussian<T>& other) const {
    const Impl& x = other.impl();
    return {eta + x.eta, lambda + x.lambda, lm + x.lm};
  }

  CanonicalGausssian<T> divide(const CanonicalGaussian<T>& other) const {
    const Impl& x = other.impl();
    return {eta - x.eta, lambda - x.lambda, lm - x.lm};
  }

  CanonicalGaussian<T> weighted_update(const CanonicalGaussian& other, const Val<T>& a) const {
    const Impl& x = other.impl();
    T b = 1 - a;
    return {b * eta + a * x.eta, b * lambda + a * x.lambda, b * lm + a * x.lm};
  }

  void multiply_in(const CanonicalGaussian<T>& other, const Dims& dims) {
    const Impl& x = other.impl();
    sub(eta, dims) += x.eta;
    sub(lambda, dims, dims) += x.lambda;
    lm += x.lm;
  }

  void multiply_in(const CanonicalGaussian<T>& other, unsigned start, unsigned n) {
    const Impl& x = other.impl();
    eta.segment(start, n) += x.eta;
    lambda.block(start, start, n, n) += x.lambda;
    lm += x.lm;
  }

  void divide_in(const CanonicalGaussian<T>& other, const Dims& dims) {
    const Impl& x = other.impl();
    sub(eta, dims) -= x.eta;
    sub(lambda, dims, dims) -= x.lambda;
    lm -= x.lm;
  }

  void divide_in(const CanonicalGaussian<T>& other, unsigned start, unsigned n) {
    const Impl& x = other.impl();
    eta.segment(start, n) -= x.eta;
    lambda.block(start, start, n, n) -= x.lambda;
    lm -= x.lm;
  }

  CanonicalGaussian<T> multiply(const CanonicalGaussian<T>& other, const Dims& i, const Dims& j)
  const {
    const Impl& x = other.impl();
    std::unique_ptr<Impl> result = zeros((i | j).count());
    sub(result->eta, i) = eta;
    sub(result->eta, j) += x.eta;
    sub(result->lambda, i, i) = lambda;
    sub(result->lambda, j, j) += x.lambda;
    result->lm = lm + x.lm;
    return std::move(result);
  }

  CanonicalGaussian<T> multiply(const CanonicalGaussian<T>& other, unsigned si, unsigned sj) const {
    const Impl& x = other.impl();
    size_t m = size(), n = x.size();
    std::unique_ptr<Impl> result = zeros(std::max(si + m, sj + n));
    result->eta.segment(si, m) = eta;
    result->eta.segment(sj, n) += x.eta;
    result->lambda.block(si, si, m, m) = lambda;
    result->lambda.block(sj, sj, n, n) += x.lambda;
    result->lm = lm + x.lm;
    return std::move(result);
  }

  CanonicalGaussian<T> divide(const CanonicalGaussian<T>& other, const Dims& i, const Dims& j)
  const {
    const Impl& x = other.impl();
    std::unique_ptr<Impl> result = zeros((i | j).count());
    sub(result->eta, i) = eta;
    sub(result->eta, j) -= x.eta;
    sub(result->lambda, i, i) = lambda;
    sub(result->lambda, j, j) -= x.lambda;
    result->lm = lm - x.lm;
    return std::move(result);
  }

  CanonicalGaussian<T> divide(const CanonicalGaussian<T>& other, unsigned si, unsigned sj) const {
    const Impl& x = other.impl();
    size_t m = size(), n = x.size();
    std::unique_ptr<Impl> result = zeros(std::max(si + m, sj + n));
    result->eta.segment(si, m) = eta;
    result->eta.segment(sj, n) -= x.eta;
    result->lambda.block(si, si, m, m) = lambda;
    result->lambda.block(sj, sj, n, n) -= x.lambda;
    result->lm = lm - x.lm;
    return std::move(result);
  }

  Exp<T> marginal() const {
    CholeskyType chol(lambda);
    T lv = lm +
      (+ size() * std::log(two_pi<RealType>())
        - logdet(chol)
        + eta.dot(chol.solve(eta))) / T(2);
    return {lv, log_tag()};
  }

  CanonicalGaussian<T> marginal(const Dims& i) const {
    return reduce(i).marginal();
  }

  CanonicalGaussian<T> marginal_front(unsigned n) const {
    return reduce(n, 0, n).marginal();
  }

  CanonicalGaussian<T> marginal_back(unsigned n) const {
    return reduce(n, size() - n, 0).marginal();
  }

  Exp<T> maximum(Assignment* a) const {
    CholeskyType chol(lambda);
    if (chol.info() == Eigen::Success) {
      VectorType vec = chol.solve(eta);
      T value = lm + eta.dot(vec) / 2;
      if (a) *a = std::move(vec);
      return {value, log_tag()};
    };

    throw NumericalError(
      "CanonicalGaussian::maximum(): Cholesky decomposition failed"
    );
  }

  CanonicalGaussian<T> maximum(const Dims& i) const {
    return reduce(i).maximum();
  }

  CanonicalGaussian<T> maximum_front(unsigned n) const {
    return reduce(n, 0, n).maximum();
  }

  CanonicalGaussian<T> maximum_back(unsigned n) const {
    return reduce(n, size() - n, 0).maximum();
  }

  CanonicalGaussian<T> restrict(const Dims& i, const Assignment& a) const {
    return reduce(i).restrict(a);
  }

  CanonicalGaussian<T> restrict_front(unsigned n, const Assignment& a) const {
    return reduce(n, 0, n).restrict(a);
  }

  CanonicalGaussian<T> restrict_back(unsigned n, const Assignment& a) const {
    return reduce(n, size() - n, 0).restrict(a);
  }

  CanonicalGaussian<T> conditional(const Dims& i) const {
    return reduce(i).conditinal();
  }

  CanonicalGaussian<T> conditional_front(unsigned n) const {
    return reduce(n, 0, n).conditional();
  }

  CanonicalGaussian<T> conditional_back(unsigend n) const {
    return reduce(n, size() - n, 0).conditional(vec);
  }

  T entropy() const {
    CholeskyType chol;
    chol.compute(lambda);
    return (size() * (std::log(two_pi<T>()) + T(1)) - logdet(chol)) / T(2);
  }

  T kl_divergence(const CanonicalGaussian<T>& other) const {
    const Impl& p = *this;
    const Impl& q = other.impl();
    assert(p.size() == q.size());
    CholeskyType chol_p, chol_q;
    chol_p.compute(p.lambda);
    chol_q.compute(q.lambda);
    VectorType diff = chol_p.solve(p.eta) - chol_q.solve(q.eta);
    auto identity = MatrixType::Identity(p.size(), p.size());
    T trace = (q.lambda.array() * chol_p.solve(identity).array()).sum();
    T means = diff.transpose() * q.lambda * diff;
    T logdets = logdet(chol_p) - logdet(chol_q);
    return (trace + means + logdets - p.size()) / T(2);
  }

  T max_diff(const CanonicalGaussian& other) const {
    const Impl& x = other.impl();
    T d_eta = (eta - x.eta).array().abs().maxCoeff();
    T d_lam = (lambda - x.lambda).array().abs().maxCoeff();
    return std::max(d_eta, d_lam);
  }

  bool normalizable() const {
    return std::isfinite(marginal());
  }

  void normalize() {
    lm -= marginal().lv;
  }

  CanonicalGaussian<T> reorder(const IndexVector) {
    ///
  }

  template <typename MAT, typename VEC>
  struct Reduce {
    VEC eta_x, eta_y;
    MAT lam_xx, lam_xy, lamy_yy;
    T lm;

    CanonicalGaussian<T> collapse(bool adjust_lm) {
      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      Eigen::LLT<MatrixType> chol_yy;
      MatrixType sol_yx = lam_xy.transpose();
      VectorType sol_y = eta_y;
      chol_yy.compute(lam_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error("CanonicalGaussian collapse: Cholesky decomposition failed");
      }
      if (!eta_y.empty()) {
        chol_yy.solveInPlace(sol_yx);
        chol_yy.solveInPlace(sol_y);
      }

      // compute the aggregate parameters
      auto result = std::make_unique<Impl>();
      result->eta = std::move(eta_x);
      result->eta.noalias() -= sol_yx.transpose() * eta_y;
      result->lambda = std::move(lam_xx);
      result->lambda.noalias() -= lam_xy * sol_yx;
      result->lm = lm + RealType(0.5) * eta_y.dot(sol_y);

      // adjust the log-multiplier if computing the marginal
      if (adjust_lm) {
        result->lm += RealType(0.5) * std::log(two_pi<RealType>()) * eta_y.size();
        result->lm -= RealType(0.5) * logdet(chol_yy);
      }

      return std::move(result);
    }

    CanonicalGaussian<T> marginal() {
      return collapse(/*adjust_lm=*/true);
    }

    CanonicalGaussian<T> maximum() {
      return collapse(/*adjust_lm=*/false);
    }

    CanonicalGaussian<T> conditional() {
      // FIX THE INDICES

      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      Eigen::LLT<MatrixType> chol_yy;
      MatrixType sol_yx = lam_xy.transpose();
      VectorType sol_y = eta_y;
      chol_yy.compute(lam_yy);
      if (chol_yy.info() != Eigen::Success) {
        throw numerical_error("canonical_gaussian conditional: Cholesky decomposition failed");
      }
      if (!eta_y.empty()) {
        chol_yy.solveInPlace(sol_yx);
        chol_yy.solveInPlace(sol_y);
      }

      // compute the conditional parameters
      unsigned m = eta_x.size();
      unsigned n = eta_y.size();
      auto result = std::make_unique<Impl>(m + n);
      result->eta.segment(0, m) = eta_x;
      result->eta.segment(m, n).noalias() = sol_yx.transpose() * eta_x;
      result->lambda.block(0, 0, m, m) = lam_xx;
      result->lambda.block(0, m, m, n) = lam_xy;
      result->lambda.block(m, 0, n, m) = lam_xy.transpose();
      result->lambda.block(m, m, n, n).noalias() = sol_yx.transpose() * lam_xy;
      result->lm = T(0.5) * (logdet(chol_yy) - std::log(two_pi<RealType>()) * m - eta_x.dot(sol_y));

      return std::move(result);
    }

    CanonicalGaussian<T> restrict(const Assignment& a) {
      auto values = a.vector<0, T>();
      auto result = std::make_unique<Impl>();
      result->eta = std::move(eta_x);
      result->eta.noalias() -= lam_xy * values;
      result->lambda = std::move(lam_xx);
      result->lm = lm + eta_y.dot(values) - T(0.5) * values.transpose() * lam_yy * values;
      return result;
    }
  };

  Reduce<MatrixType, VectorType> reduce(const Dims& i) const {
    Dims j = exclude(i);
    return {
      sub(eta, i),
      sub(eta, j),
      sub(lambda, i, i),
      sub(lambda, i, j),
      sub(lambda, j, j),
      lm,
    };
  }

  Reduce<SegmentType, BlockType> reduce(unsigned nx, unsigned sx, unsigned sy) const {
    unsigned ny = size() - nx;
    return {
      eta.segment(sx, nx),
      eta.segment(sy, ny),
      lambda.block(sx, sx, nx, nx),
      lambda.block(sx, sy, nx, ny),
      lambda.block(sy, sy, ny, ny),
      lm,
    };
  }
}; // class Impl

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(const CanonicalGaussian& other) {
  impl_.reset(new Impl<CanonicalGausssian>>(other.impl_));
}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Exp<T> value) {
  impl_.reset(new Impl<CanonicalGaussian>>());
  impl_->lm = value.lv;
}

template <typename T>
CanonicalGaussian::CanonicalGaussian(const MomentGaussian& mg) {
  impl_.reset(new Impl<CanonicalGaussian>>());
  impl_->assign(mg);
}

template <typename T>
CanonicaLGaussian<T>::CanonicalGaussian(const VectorType& eta, const MatrixType& lambda, T lv) {
  impl_.reset(new Impl<CanonicalGaussian>{eta, lambda, lv});
}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(VectorType&& eta, MatrixType&& lambda, T lv) {
  impl_.reset(new Impl<CanonicalGaussian>{std::move(eta), std::move(lambda), lv});
}

template <typename T>
CanonicalGaussian<T>::~CanonicalGaussian() {}

template <typename T>
T CanonicalGaussian<T>::log_multiplier() const {
  return param_->lm;
}

const VectorType& CanonicalGuassian<T>::inf_vector() const {
  return param_->eta;
}

const MatrixType& CanonicalGuassian<T>::inf_matrix() const {
  return param_->lambda;
}

}  // namespace libgm
