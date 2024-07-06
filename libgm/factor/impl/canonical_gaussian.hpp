#include "../canonical_gaussian.hpp"

#include <libgm/factor/moment_gaussian.hpp>

#include <boost/math/constants/constants.hpp>

namespace libgm {

template <typename T>
struct CanonicalGausssian<T>::Impl {
  /// The type of the LLT Cholesky decomposition object.
  using CholeskyType = Eigen::LLT<DenseMatrix<T>>;

  /// The partitioning of the random vector.
  Shape shape;

  /// The information vector.
  DenseVector<T> eta;

  /// The information matrix.
  DenseMatrix<T> lambda;

  /// The log-multiplier.
  T lm = T(0);

  /// Pre-computed constant.
  static const T log_two_pi = std::log(boost::math::constants::two_pi<T>);

  /// Returns the spans for the given dimensions.
  std::vector<Span> spans(const Dims& dims, bool ignore_out_of_range = false) const {
    return shape.spans(dims, ignore_out_of_range);
  }

  // Evaluation
  //--------------------------------------------------------------------------
  T log(const Values& values) const {
    auto x = values.vector<T>();
    return -T(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
  }

  // Conversion
  //--------------------------------------------------------------------------
  MomentGaussian<T> moment() const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "moment");
    DenseVector<T> mean = chol.solve(cg.eta);
    T adjust = (size() * log_two_pi - logdet(chol) + mean.dot(eta)) / T(2);
    return {shape, mean, chol.solve(DenseMatrix<T>::Identity(n, n)), lm + adjust};
  }

  // Object operations
  //--------------------------------------------------------------------------

  void equals(const Object& other) const override {
    const Impl& x = impl(other);
    return shape == x.shape && eta == x.eta && lambda == x.lambda && lm == other.lm;
  }

  void print(std::ostream& out) const override {
    out << shape << std::endl << eta << std::endl << lambda << std::endl << lm;
  }

  void save(oarchive& ar) const override {
    ar << shape << eta << lambda << lm;
  }

  /// Deserializes the parameters from an archive.
  void load(iarchive& ar) override {
    ar >> shape >> eta >> lambda >> lm;
  }

  // Assignment
  //--------------------------------------------------------------------------

  void assign(const Exp<T>& value) {
    shape.clear();
    eta.clear();
    lambda.claer();
    lm = value.lv;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  ImplPtr multiply(const Exp<T>& x) const {
    return std::make_unique<Impl>(shape, eta, lambda, lm + log(x));
  }

  ImplPtr divide(const Exp<T>& x) const {
    return std::make_unique<Impl>(shape, eta, lambda, lm - log(x));
  }

  ImplPtr divide_inverse(const Exp<T>& x) const {
    return std::make_unique<Impl>(shape, -eta, -lambda, log(x) - lm);
  }

  ImplPtr multiply(const Object& other) const {
    const Impl& x = impl(other);
    check(x.shape);
    return std::make_unique<Impl>(shape, eta + x.eta, lambda + x.lambda, lm + x.lm);
  }

  ImplPtr divide(const Object& other) const {
    const Impl& x = impl(other);
    check(x.shape);
    return std::make_unique<Impl>(shape, eta - x.eta, lambda - x.lambda, lm - x.lm);
  }

  void multiply_in(const Exp<T>& x) {
    lm += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    lm -= x.lv;
  }

  void multiply_in(const Object& other) {
    const Impl& x = impl(other);
    check(x.shape);
    eta += x.eta;
    lambda += x.lambda;
    lm += x.lm;
  }

  void divide_in(const Object& other) {
    const Impl& x = impl(other);
    check(x.shape);
    eta -= x.eta;
    lambda -= x.lambda;
    lm -= x.lm;
  }

  // Join operations
  //--------------------------------------------------------------------------

  ImplPtr multiply_front(const Object& other) const {
    std::unique_ptr<Impl> result(new Impl(*this));
    result->multiply_in_front(other);
    return result;
  }

  ImplPtr multiply_back(const Object& other) const {
    std::unique_ptr<Impl> result(new Impl(*this));
    result->multiply_in_back(other);
    return result;
  }

  ImplPtr multiply(const Object& other, const Dims& i, const Dims& j) const {
    const Impl& x = impl(other);
    auto result = std::make_unique<Impl>(join(shape, x.shape, i, j));
    std::vector<Span> is = result->spans(i);
    std::vector<Span> js = result->spans(j);
    sub(result->eta, is) = eta;
    sub(result->eta, js) += x.eta;
    sub(result->lambda, is, is) = lambda;
    sub(result->lambda, js, js) += x.lambda;
    result->lm = lm + x.lm;
    return std::move(result);
  }

  ImplPtr divide(const Object& other, const Dims& i, const Dims& j) const {
    const Impl& x = impl(other);
    auto result = std::make_unique<Impl>(join(shape, x.shape, i, j));
    std::vector<Span> is = result->spans(i);
    std::vector<Span> js = result->spans(j);
    sub(result->eta, is) = eta;
    sub(result->eta, js) -= x.eta;
    sub(result->lambda, is, is) = lambda;
    sub(result->lambda, js, js) -= x.lambda;
    result->lm = lm - x.lm;
    return std::move(result);
  }

  void multiply_in_front(const Object& other) {
    const Impl& x = impl(other);
    check_front(x.shape);
    size_t n = x.size();
    eta.head(n) += x.eta;
    lambda.topLeftCorner(n, n) += x.lambda;
    lm += x.lm;
  }

  void multiply_in_back(const Object& other) {
    const Impl& x = impl(other);
    check_back(x.shape);
    size_t n = x.size();
    eta.tail(n) += x.eta;
    lambda.bottomRightCorner(n, n) += x.lambda;
    lm += x.lm;
  }

  void multiply_in(const Object& other, const Dims& dims) {
    const Impl& x = impl(other);
    check(dims, x.shape);
    std::vector<Span> is = spans(dims);
    sub(eta, is) += x.eta;
    sub(lambda, is, is) += x.lambda;
    lm += x.lm;
  }

  void divide_in_front(const Object& other) {
    const Impl& x = impl(other);
    check_front(x.shape);
    size_t n = s.size();
    eta.head(n) -= x.eta;
    lambda.topLeftCorner(n, n) -= x.lambda;
    lm -= x.lm;
  }

  void divide_in_back(const Object& other) {
    const Impl& x = impl(other);
    check_back(x.shape);
    size_t n = s.size();
    eta.tail(n) -= x.eta;
    lambda.bottomRightCorner(n, n) -= x.lambda;
    lm -= x.lm;
  }

  void divide_in(const Object& other, const Dims& dims) {
    const Impl& x = impl(other);
    check(dims, x.shape);
    std::vector<Span> is = spans(dims);
    sub(eta, i) -= x.eta;
    sub(lambda, i, i) -= x.lambda;
    lm -= x.lm;
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  ImlPtr pow(T alpha) const {
    return std::make_unique<Impl>(shape, eta * alpha, lambda * alpha, lm * alpha);
  }

  ImplPtr weighted_update(const Object& other, T a) const {
    const Impl& x = impl(other);
    assert(shape == other.shape);
    T b = 1 - a;
    return std::make_unique<Impl>(shape, b*eta + a*x.eta, b*lambda + a*x.lambda, b*lm + a*x.lm);
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "marginal");
    T lv = lm + ((size() * log_two_pi) - logdet(chol) + eta.dot(chol.solve(eta))) / T(2);
    return Exp<T>(lv);
  }

  Exp<T> maximum(Values* values) const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "maximum");
    DenseVector<T> vec = chol.solve(eta);
    if (values) *values = vec;
    return Exp<T>(lm + eta.dot(vec) / T(2));
  }

  ImplPtr marginal_front(unsigned n) const {
    return reduce_front(n).marginal();
  }

  ImplPtr marginal_back(unsigned n) const {
    return reduce_back(n).marginal();
  }

  ImplPtr marginal(const Dims& dims) const {
    return reduce(dims).marginal();
  }

  ImplPtr maximum_front(unsigned n) const {
    return reduce_front(n).maximum();
  }

  ImplPtr maximum_back(unsigned n) const {
    return reduce_back(n).maximum();
  }

  ImplPtr maximum(const Dims& dims) const {
    return reduce(dims).maximum();
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    lm -= marginal().lv;
  }

  void normalize(unsigned n) {
    // determine the block sizes
    size_t nx = shape.sum_head(n);
    size_t ny = size() - nx;

    // compute sol_xy = lam_xx^{-1} * lam_xy and sol_x = lam_xx^{-1} eta_x
    CholeskyType chol_xx;
    DenseMatrix<T> sol_xy = lambda.topRightCorner(nx, ny);
    DenseVector<T> sol_x = eta.head(nx);
    chol_xx.compute(lam.bottomRightCorner(nx, nx));
    check(chol_xx, "conditional");
    if (nx) {
      chol_xx.solveInPlace(sol_xy);
      chol_xx.solveInPlace(sol_x);
    }

    // update the distribution
    eta.tail(ny).noalias() = sol_xy.transpose() * eta.head(nx);
    lambda.bottomRightCorner(ny, ny).noalias() = lambda.bottomLeftCorner(ny, nx) * sol_xy;
    lm = (logdet(chol_xx) - log_two_pi * nx - eta.head(nx).dot(sol_x)) / T(2);

    return std::move(result);
  }

  // Restrictions
  //--------------------------------------------------------------------------

  ImplPtr restrict_front(const Values& values) const {
    return reduce_front(values.size()).restrict(values);
  }

  ImplPtr restrict_back(const Values& values) const {
    return reduce_back(values.size()).restrict(values);
  }

  ImplPtr restrict(const Dims& dims, const Values& values) const {
    return reduce(dims).restrict(values);
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "entropy");
    return (size() * (log_two_pi + T(1)) - logdet(chol)) / T(2);
  }

  T kl_divergence(const Object& other) const {
    const Impl& p = *this;
    const Impl& q = impl(other);
    check(q.shape);
    CholeskyType chol_p, chol_q;
    chol_p.compute(p.lambda);
    chol_q.compute(q.lambda);
    check(chol_p, "kl_divergence (p)");
    check(chol_q, "kl_divergence (q)");
    DenseVector<T> diff = chol_p.solve(p.eta) - chol_q.solve(q.eta);
    auto identity = DenseMatrix<T>::Identity(p.size(), p.size());
    T trace = (q.lambda.array() * chol_p.solve(identity).array()).sum();
    T means = diff.transpose() * q.lambda * diff;
    T logdets = logdet(chol_p) - logdet(chol_q);
    return (trace + means + logdets - p.size()) / T(2);
  }

  T max_diff(const Object& other) const {
    const Impl& x = impl(other);
    check(x.shape);
    T d_eta = (eta - x.eta).array().abs().maxCoeff();
    T d_lam = (lambda - x.lambda).array().abs().maxCoeff();
    T d_lm = lm - x.lm;
    return std::max({d_eta, d_lam, d_lm});
  }

  // Implementation of reductions
  //--------------------------------------------------------------------------

  template <typename MAT, typename VEC>
  struct Reduce {
    Shape shape;
    VEC eta_x, eta_y;
    MAT lam_xx, lam_xy, lamy_yy;
    T lm;

    ImplPtr collapse(bool adjust_lm) {
      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      Eigen::LLT<DenseMatrix<T>> chol_yy;
      DenseMatrix<T> sol_yx = lam_xy.transpose();
      DenseVector<T> sol_y = eta_y;
      chol_yy.compute(lam_yy);
      check(chol_yy, "collapse");
      if (!eta_y.empty()) {
        chol_yy.solveInPlace(sol_yx);
        chol_yy.solveInPlace(sol_y);
      }

      // compute the aggregate parameters
      auto result = std::make_unique<Impl>();
      result->shape = std::move(shape);
      result->eta = std::move(eta_x);
      result->eta.noalias() -= sol_yx.transpose() * eta_y;
      result->lambda = std::move(lam_xx);
      result->lambda.noalias() -= lam_xy * sol_yx;
      result->lm = lm + eta_y.dot(sol_y) / T(2);

      // adjust the log-multiplier if computing the marginal
      if (adjust_lm) {
        result->lm += (log_two_pi * eta_y.size() - logdet(chol_yy)) / T(2);
      }

      return std::move(result);
    }

    ImplPtr marginal() {
      return collapse(/*adjust_lm=*/true);
    }

    ImplPtr maximum() {
      return collapse(/*adjust_lm=*/false);
    }

    ImplPtr restrict(const Values& values) {
      auto values = a.vector<T>();
      auto result = std::make_unique<Impl>();
      result->shape = std::move(shape);
      result->eta = std::move(eta_x);
      result->eta.noalias() -= lam_xy * values;
      result->lambda = std::move(lam_xx);
      result->lm = lm + eta_y.dot(values) - T(0.5) * values.transpose() * lam_yy * values;
      return result;
    }
  };

  Reduce<SegmentType, BlockType> reduce_front(unsigned n) const {
    size_t nx = shape.prefix_sum(n);
    size_t ny = size() - nx;
    return {
      shape.prefix(n),
      eta.head(nx),
      eta.tail(ny),
      lambda.topLeftCorner(nx, nx),
      lambda.topRightCorner(nx, ny),
      lambda.bottomRightCorner(ny, ny),
      lm,
    };
  }

  Reduce<SegmentType, BlockType> reduce_back(unsigned n) const {
    size_t nx = shape.suffix_sum(n);
    size_t ny = size() - nx;
    return {
      shape.suffix(n),
      eta.tail(nx),
      eta.head(ny),
      lambda.bottomRightCorner(nx, nx),
      lambda.bottomLeftCorner(nx, ny),
      lambda.topLeftCorner(ny, ny),
      lm,
    };
  }

  Reduce<DenseMatrix<T>, DenseVector<T>> reduce(const Dims& i) const {
    std::vector<Span> is = shape.spans(i);
    std::vector<Span> js = shape.spans(~i, /*ignore_out_of_range=*/true);
    return {
      shape.subseq(i),
      sub(eta, is),
      sub(eta, js),
      sub(lambda, is, is),
      sub(lambda, is, js),
      sub(lambda, js, js),
      lm,
    };
  }
}; // class Impl

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Exp<T> value)
  : Factor(std::make_unique<Impl>({}, value)) {}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Shape shape, Exp<T> value)
  : Factor(std::make_unique<Impl>(std::move(shape), value)) {}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Shape shape, DenseVector<T> eta, DenseMatrix<T> lambda, T lv)
  : Factor(std::make_unique<Impl>(std::move(shape), std::move(eta), std::move(lambda), lv)) {}

template <typename T>
unsigned CanonicalGaussian<T>::arity() const {
  return impl().shape.size();
}

template <typename T>
T CanonicalGaussian<T>::log_multiplier() const {
  return impl().lm;
}

template <typename T>
const DenseVector<T>& CanonicalGuassian<T>::inf_vector() const {
  return impl().eta;
}

template <typename T>
const DenseMatrix<T>& CanonicalGuassian<T>::inf_matrix() const {
  return impl().lambda;
}

template <typename T>
Exp<T> CanonicalGaussian<T>::operator()(const Values& values) const {
  return Exp<T>(log(a));
}

template <typename T>
T CanonicalGaussian<T>::log(const Values& values) const {
  return impl().log(values);
}

template <typename T>
MomentGaussian<T> CanonicalGaussian<T>::moment() const {
  return impl().moment();
}

} // namespace libgm
