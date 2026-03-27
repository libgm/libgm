  #include <boost/math/constants/constants.hpp>

#include "../moment_gaussian.hpp"
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>

namespace libgm {

template <typename T>
struct MomentGaussian<T>::Impl {
  /// Helper types
  using CholeskyType = Eigen::LLT<Matrix<T>>;
  using SegmentType = decltype(std::declval<const Vector<T>>().head(size_t(0)));
  using BlockType = decltype(std::declval<const Matrix<T>>().topLeftCorner(size_t(0), size_t(0)));

  /// The shape of the head + tail arguments.
  Shape shape;

  /// The conditional mean.
  Vector<T> mean;

  /// The covariance matrix.
  Matrix<T> cov;

  /// The coefficient matrix.
  Matrix<T> coef;

  /// The log-multiplier.
  T lm = T(0);

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  explicit Impl(Exp<T> value)
    : lm(value.lv) {}

  explicit Impl(Shape shape)
    : shape(std::move(shape)) {
    size_t m = this->shape.sum();
    mean.resize(m);
    cov.resize(m, m);
    coef.resize(m, 0);
  }

  Impl(Shape shape, Vector<T> mean, Matrix<T> cov, T lm = T(0))
    : Impl(std::move(shape), mean, std::move(cov), Matrix<T>(mean.size(), 0), lm) {}

  Impl(Shape shape, Vector<T> mean, Matrix<T> cov, Matrix<T> coef, T lm = T(0))
    : shape(std::move(shape)),
      mean(std::move(mean)),
      cov(std::move(cov)),
      coef(std::move(coef)),
      lm(lm) {
    assert(this->mean.size() == this->cov.rows());
    assert(this->mean.size() == this->cov.cols());
    assert(this->mean.size() == this->coef.rows());
    assert(this->mean.size() + this->coef.cols() == this->shape.sum());
  }

  /// Pre-computed constant.
  static const T log_two_pi;

  /// Checks if the cholesky decomposition succeeded.
  static void check(const CholeskyType& chol, const char* method) {
    if (chol.info() != Eigen::Success) {
      throw std::runtime_error(
        "MomentGaussian::" + std::string(method) + ": Cholesky decomposition failed");
    }
  }

  // Accessors
  //--------------------------------------------------------------------------

  unsigned arity() const {
    return shape.size();
  }

  size_t size() const {
    return head_size() + tail_size();
  }

  size_t head_size() const {
    return coef.rows();
  }

  size_t tail_size() const {
    return coef.cols();
  }

  bool is_marginal() const {
    return tail_size() == 0;
  }

  // Object operations
  //--------------------------------------------------------------------------

  void print(std::ostream& out) const {
   out << shape << std::endl
       << mean << std::endl
       << cov << std::endl
       << coef << std::endl
       << lm;
  }

  // Join operations
  //--------------------------------------------------------------------------
  std::unique_ptr<Impl> multiply_head_tail(const Impl& other) const {
    size_t m = head_size();
    size_t n = tail_size();
    assert(other.head_size() == n);
    assert(other.tail_size() == 0);
    auto result = std::make_unique<Impl>(shape);
    Matrix<T> cov_mn = coef * other.cov;
    result->mean.tail(n) = other.mean;
    result->mean.head(m) = mean + coef * other.mean;
    result->cov.bottomRightCorner(n, n) = other.cov;
    result->cov.bottomLeftCorner(n, m) = cov_mn.transpose();
    result->cov.topRightCorner(m, n) = cov_mn;
    result->cov.topLeftCorner(m, m) = cov_mn * coef.transpose() + cov;
    result->lm = lm + other.lm;
    return result;
  }

  std::unique_ptr<Impl> multiply_dims(const Impl& other, const Dims& i, const Dims& j) const {
    assert(is_marginal());
    assert(other.is_marginal());
    assert((i & j).none() && "MomentGaussian::multiply does not support index overlap");
    auto result = std::make_unique<Impl>(join(shape, other.shape, i, j));
    result->cov.setZero();
    Spans is = result->shape.spans(i);
    Spans js = result->shape.spans(j);
    sub(result->mean, is) = mean;
    sub(result->mean, js) = other.mean;
    sub(result->cov, is, is) = cov;
    sub(result->cov, js, js) = other.cov;
    result->lm = lm + other.lm;
    return result;
  }

  // Aggregates
  //--------------------------------------------------------------------------

  Exp<T> marginal() const {
    return Exp<T>(lm);
  }

  T log(const Vector<T>& values) const {
    size_t m = head_size();
    size_t n = tail_size();
    assert(values.size() == m + n);
    Vector<T> residual = values.head(m) - mean;
    if (n > 0) {
      residual.noalias() -= coef * values.tail(n);
    }
    CholeskyType chol;
    chol.compute(cov);
    check(chol, "log");
    return lm - (m * log_two_pi + logdet(chol) + residual.dot(chol.solve(residual))) / T(2);
  }

  Exp<T> maximum(Vector<T>* values) const {
    if (values) *values = mean;
    CholeskyType chol;
    chol.compute(cov);
    check(chol, "maximum");
    return Exp<T>((-log_two_pi * head_size() - logdet(chol)) / T(2) + lm);
  }

  std::unique_ptr<Impl> marginal_front(unsigned n) const {
    assert(is_marginal());
    assert(n <= head_size());
    return std::make_unique<Impl>(
      shape.prefix(n),
      mean.head(n),
      cov.topLeftCorner(n, n),
      lm
    );
  }

  std::unique_ptr<Impl> marginal_back(unsigned n) const {
    assert(n >= tail_size());
    unsigned m = n - tail_size();
    assert(m <= head_size());
    return std::make_unique<Impl>(
      shape.suffix(n),
      mean.tail(m),
      cov.bottomRightCorner(n, n),
      coef.bottomRows(n),
      lm
    );
  }

  std::unique_ptr<Impl> marginal_dims(const Dims& i) const {
    assert(is_marginal());
    Spans is = shape.spans(i);
    return std::make_unique<Impl>(
      shape.select(i),
      sub(mean, is),
      sub(cov, is, is),
      lm
    );
  }

  std::unique_ptr<Impl> maximum_front(unsigned n) const {
    return marginal_to_maximum(marginal_front(n));
  }

  std::unique_ptr<Impl> maximum_back(unsigned n) const {
    return marginal_to_maximum(marginal_back(n));
  }

  std::unique_ptr<Impl> maximum_dims(const Dims& dims) const {
    return marginal_to_maximum(marginal_dims(dims));
  }

  std::unique_ptr<Impl> marginal_to_maximum(std::unique_ptr<Impl> result) const {
    result->lm += maximum(nullptr).lv - result->maximum(nullptr).lv;
    return result;
  }

  // Restriction
  //--------------------------------------------------------------------------

  /// Implements the restrict operation for marginals
  template <typename MAT, typename VEC>
  struct RestrictMarginal {
    Shape shape;
    // retained: x, restricted head: y
    VEC mean_x, mean_y;
    MAT cov_xx, cov_xy, cov_yy;
    T lm;

    std::unique_ptr<Impl> operator()(const Vector<T>& values) {
      // compute sol_yx = cov_yy^{-1} cov_yx using Cholesky decomposition
      Matrix<T> sol_yx = cov_xy.transpose();
      CholeskyType chol_yy;
      chol_yy.compute(cov_yy);
      check(chol_yy, "restrict");
      if (mean_y.size() != 0) {
        chol_yy.solveInPlace(sol_yx);
      }

      // compute the residual over y (observation vec_y - the prediction)
      Vector<T> res_y = values - mean_y;

      // compute the output
      auto r = std::make_unique<Impl>();
      r->shape = std::move(shape);
      r->mean = mean_x + sol_yx.transpose() * res_y;
      r->cov = cov_xx - cov_xy * sol_yx;
      r->lm = lm - (mean_y.size() * log_two_pi + logdet(chol_yy) + res_y.dot(chol_yy.solve(res_y))) / T(2);
      r->coef.resize(r->mean.size(), 0);
      return r;
    }
  };

  std::unique_ptr<Impl> restrict_front(const Vector<T>& values) const {
    assert(is_marginal() && values.size() <= head_size());
    size_t m = values.size();
    size_t n = head_size() - m;
    unsigned new_arity = arity() - shape.prefix_size(m);
    return RestrictMarginal<BlockType, SegmentType>{
      shape.suffix(new_arity),
      mean.tail(n),
      mean.head(m),
      cov.bottomRightCorner(n, n),
      cov.bottomLeftCorner(n, m),
      cov.topLeftCorner(m, m),
      lm,
    }(values);
  }

  std::unique_ptr<Impl> restrict_back(const Vector<T>& values) const {
    assert(is_marginal() && values.size() <= head_size());
    size_t m = head_size() - values.size();
    size_t n = values.size();
    unsigned new_arity = arity() - shape.suffix_size(n);
    return RestrictMarginal<BlockType, SegmentType>{
      shape.prefix(new_arity),
      mean.head(m),
      mean.tail(n),
      cov.topLeftCorner(m, m),
      cov.topRightCorner(m, n),
      cov.bottomRightCorner(n, n),
      lm,
    }(values);
  }

  std::unique_ptr<Impl> restrict_dims(const Dims& i, const Vector<T>& values) const {
    assert(is_marginal());
    Spans is = shape.spans(i);
    Spans js = shape.spans(~i, /*ignore_out_of_range=*/true);
    return RestrictMarginal<Matrix<T>, Vector<T>>{
      shape.omit(i),
      sub(mean, js),
      sub(mean, is),
      sub(cov, js, js),
      sub(cov, js, is),
      sub(cov, is, is),
      lm,
    }(values);
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    assert(is_marginal());
    CholeskyType chol;
    chol.compute(cov);
    return (size() * (log_two_pi + T(1)) + logdet(chol)) / T(2);
  }

  T kl_divergence(const MomentGaussian& other) const {
    const Impl& p = *this;
    const Impl& q = other.impl();
    assert(p.is_marginal() && q.is_marginal());
    assert(p.shape == q.shape);
    unsigned m = p.head_size();
    CholeskyType chol_p(p.cov);
    CholeskyType chol_q(q.cov);
    auto identity = Matrix<T>::Identity(m, m);
    T trace = (p.cov.array() * chol_q.solve(identity).array()).sum();
    T means = (p.mean - q.mean).transpose() * chol_q.solve(p.mean - q.mean);
    T logdets = -logdet(chol_p) + logdet(chol_q);
    return (trace + means + logdets - m) / 2;
  }

  T max_diff(const MomentGaussian<T>& other) const {
    const Impl& x = other.impl();
    T diff_mean = (mean - x.mean).array().abs().maxCoeff();
    T diff_cov  = (cov - x.cov).array().abs().maxCoeff();
    T diff_coef = (coef - x.coef).array().abs().maxCoeff();
    return std::max({diff_mean, diff_cov, diff_coef});
  }

  // Conversion
  //--------------------------------------------------------------------------

  CanonicalGaussian<T> canonical() const {
    CholeskyType chol(cov);
    check(chol, "CanonicalGaussian: Cannot invert the covariance matrix.");
    Matrix<T> sol_xy = chol.solve(coef);

    size_t m = head_size();
    size_t n = tail_size();
    size_t s = m + n;

    Vector<T> eta(s);
    eta.head(m) = chol.solve(mean);
    eta.tail(n).noalias() = -sol_xy.transpose() * mean;

    Matrix<T> lambda(s, s);
    lambda.topLeftCorner(m, m) = chol.solve(Matrix<T>::Identity(m, m));
    lambda.topRightCorner(m, n) = -sol_xy;
    lambda.bottomLeftCorner(n, m) = -sol_xy.transpose();
    lambda.bottomRightCorner(n, n).noalias() = coef.transpose() * sol_xy;

    double lv = lm - (m * log_two_pi + logdet(chol) + eta.segment(0, m).dot(mean)) / T(2);
    return CanonicalGaussian<T>(shape, std::move(eta), std::move(lambda), lv);
  }
};

template <typename T>
MomentGaussian<T>::MomentGaussian() = default;

template <typename T>
MomentGaussian<T>::MomentGaussian(const MomentGaussian& other)
  : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr) {}

template <typename T>
MomentGaussian<T>::MomentGaussian(MomentGaussian&& other) noexcept = default;

template <typename T>
MomentGaussian<T>& MomentGaussian<T>::operator=(const MomentGaussian& other) {
  if (this != &other) {
    impl_ = other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr;
  }
  return *this;
}

template <typename T>
MomentGaussian<T>& MomentGaussian<T>::operator=(MomentGaussian&& other) noexcept = default;

template <typename T>
MomentGaussian<T>::~MomentGaussian() = default;

template <typename T>
MomentGaussian<T>::MomentGaussian(std::unique_ptr<Impl> impl)
  : impl_(std::move(impl)) {}

template <typename T>
MomentGaussian<T>::MomentGaussian(Exp<T> value)
  : impl_(std::make_unique<Impl>(value)) {}

template <typename T>
MomentGaussian<T>::MomentGaussian(Shape shape)
  : impl_(std::make_unique<Impl>(std::move(shape))) {
  impl().mean.setZero();
  impl().cov.setIdentity();
  impl().lm = T(0);
}

template <typename T>
MomentGaussian<T>::MomentGaussian(Shape shape, Vector<T> mean, Matrix<T> cov, T lm)
  : impl_(std::make_unique<Impl>(std::move(shape), std::move(mean), std::move(cov), lm)) {}

template <typename T>
MomentGaussian<T>::MomentGaussian(Shape shape, Vector<T> mean, Matrix<T> cov, Matrix<T> coef, T lm)
  : impl_(std::make_unique<Impl>(std::move(shape), std::move(mean), std::move(cov), std::move(coef), lm)) {}

template <typename T>
T MomentGaussian<T>::log_multiplier() const {
  return impl().lm;
}

template <typename T>
unsigned MomentGaussian<T>::arity() const {
  return impl().arity();
}

template <typename T>
const Shape& MomentGaussian<T>::shape() const {
  return impl().shape;
}

template <typename T>
const Vector<T>& MomentGaussian<T>::mean() const {
  return impl().mean;
}

template <typename T>
const Matrix<T>& MomentGaussian<T>::covariance() const {
  return impl().cov;
}

template <typename T>
const Matrix<T>& MomentGaussian<T>::coefficients() const {
  return impl().coef;
}

template <typename T>
Exp<T> MomentGaussian<T>::operator()(const Vector<T>& values) const {
  return Exp<T>(log(values));
}

template <typename T>
T MomentGaussian<T>::log(const Vector<T>& values) const {
  return impl().log(values);
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::operator*(Exp<T> x) const {
  MomentGaussian result(*this);
  result.impl().lm += x.lv;
  return result;
}

template <typename T>
MomentGaussian<T>& MomentGaussian<T>::operator*=(Exp<T> x) {
  impl().lm += x.lv;
  return *this;
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::operator/(Exp<T> x) const {
  MomentGaussian result(*this);
  result.impl().lm -= x.lv;
  return result;
}

template <typename T>
MomentGaussian<T>& MomentGaussian<T>::operator/=(Exp<T> x) {
  impl().lm -= x.lv;
  return *this;
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::multiply_front(const MomentGaussian& other) {
  return MomentGaussian(other.impl().multiply_head_tail(impl()));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::multiply_back(const MomentGaussian& other) {
  return MomentGaussian(impl().multiply_head_tail(other.impl()));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::multiply(const MomentGaussian& other, const Dims& i, const Dims& j) const {
  return MomentGaussian(impl().multiply_dims(other.impl(), i, j));
}

template <typename T>
Exp<T> MomentGaussian<T>::marginal() const {
  return impl().marginal();
}

template <typename T>
Exp<T> MomentGaussian<T>::maximum(Vector<T>* values) const {
  return impl().maximum(values);
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::marginal_front(unsigned n) const {
  return MomentGaussian(impl().marginal_front(n));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::marginal_back(unsigned n) const {
  return MomentGaussian(impl().marginal_back(n));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::marginal_dims(const Dims& dims) const {
  return MomentGaussian(impl().marginal_dims(dims));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::maximum_front(unsigned n) const {
  return MomentGaussian(impl().maximum_front(n));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::maximum_back(unsigned n) const {
  return MomentGaussian(impl().maximum_back(n));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::maximum_dims(const Dims& dims) const {
  return MomentGaussian(impl().maximum_dims(dims));
}

template <typename T>
void MomentGaussian<T>::normalize() {
  impl().lm = T(0);
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::restrict_front(const Vector<T>& values) const {
  return MomentGaussian(impl().restrict_front(values));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::restrict_back(const Vector<T>& values) const {
  return MomentGaussian(impl().restrict_back(values));
}

template <typename T>
MomentGaussian<T> MomentGaussian<T>::restrict_dims(const Dims& dims, const Vector<T>& values) const {
  return MomentGaussian(impl().restrict_dims(dims, values));
}

template <typename T>
T MomentGaussian<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T MomentGaussian<T>::kl_divergence(const MomentGaussian& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
CanonicalGaussian<T> MomentGaussian<T>::canonical() const {
  return impl().canonical();
}

template <typename T>
MomentGaussian<T>::Impl& MomentGaussian<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *impl_;
}

template <typename T>
const typename MomentGaussian<T>::Impl& MomentGaussian<T>::impl() const {
  assert(impl_);
  return *impl_;
}

template <typename T>
const T MomentGaussian<T>::Impl::log_two_pi = std::log(boost::math::constants::two_pi<T>());

template <typename T>
std::ostream& operator<<(std::ostream& out, const MomentGaussian<T>& f) {
  return out << "MomentGaussian("
             << "shape=" << f.shape()
             << ", mean=" << f.mean()
             << ", cov=" << f.covariance()
             << ", coef=" << f.coefficients()
             << ", lm=" << f.log_multiplier()
             << ")";
}

} // namespace libgm
