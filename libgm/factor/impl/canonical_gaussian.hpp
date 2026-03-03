#include "../canonical_gaussian.hpp"

#include <libgm/factor/implements.hpp>
#include <libgm/factor/concepts.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/math/eigen/logdet.hpp>
#include <libgm/math/eigen/submatrix.hpp>
#include <libgm/math/eigen/subvector.hpp>

#include <boost/math/constants/constants.hpp>

namespace libgm {

template <typename T>
struct CanonicalGaussian<T>::Impl : Object::Impl {
  /// Helper types
  using CholeskyType = Eigen::LLT<Matrix<T>>;
  using SegmentType = decltype(std::declval<const Vector<T>>().head(size_t(0)));
  using BlockType = decltype(std::declval<const Matrix<T>>().topLeftCorner(size_t(0), size_t(0)));

  /// The partitioning of the random vector.
  Shape shape;

  /// The information vector.
  Vector<T> eta;

  /// The information matrix.
  Matrix<T> lambda;

  /// The log-multiplier.
  T lm = T(0);

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(shape, eta, lambda, lm);
  }

  // Constructors
  //--------------------------------------------------------------------------

  Impl() = default;

  explicit Impl(Exp<T> value)
    : lm(value.lv) {}

  Impl(Shape shape, Exp<T> value)
    : shape(std::move(shape)), lm(value.lv) {
    size_t n = this->shape.sum();
    eta = Vector<T>::Zero(n);
    lambda = Matrix<T>::Zero(n, n);
  }

  Impl(Shape shape, Vector<T> eta, Matrix<T> lambda, T lm)
    : shape(std::move(shape)), eta(std::move(eta)), lambda(std::move(lambda)), lm(lm) {
    assert(eta.size() == lambda.rows());
    assert(eta.size() == lambda.cols());
  }

  /// Pre-computed constant.
  static const T log_two_pi;

  /// Checks if the cholesky decomposition succeeded.
  static void check(const CholeskyType& chol, const char* method) {
    if (chol.info() != Eigen::Success) {
      throw std::runtime_error(
        "CanonicalGaussian::" + std::string(method) + ": Cholesky decomposition failed");
    }
  }

  /// Returns the spans for the given dimensions.
  Spans spans(const Dims& dims, bool ignore_out_of_range = false) const {
    return shape.spans(dims, ignore_out_of_range);
  }

  size_t size() const {
    return eta.size();
  }

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    out << shape << std::endl << eta << std::endl << lambda << std::endl << lm;
  }

  // Assignment
  //--------------------------------------------------------------------------

  void assign(const Exp<T>& value) {
    shape.clear();
    eta.resize(0);
    lambda.resize(0, 0);
    lm = value.lv;
  }

  // Direct operations
  //--------------------------------------------------------------------------

  void multiply(const Exp<T>& x, CanonicalGaussian& result) const {
    result.reset(new Impl(shape, eta, lambda, lm + x.lv));
  }

  void divide(const Exp<T>& x, CanonicalGaussian& result) const {
    result.reset(new Impl(shape, eta, lambda, lm - x.lv));
  }

  void divide_inverse(const Exp<T>& x, CanonicalGaussian& result) const {
    result.reset(new Impl(shape, -eta, -lambda, x.lv - lm));
  }

  void multiply(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    const Impl& x = other.impl();
    assert(shape == x.shape);
    result.reset(new Impl(shape, eta + x.eta, lambda + x.lambda, lm + x.lm));
  }

  void divide(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    const Impl& x = other.impl();
    assert(shape == x.shape);
    result.reset(new Impl(shape, eta - x.eta, lambda - x.lambda, lm - x.lm));
  }

  void multiply_in(const Exp<T>& x) {
    lm += x.lv;
  }

  void divide_in(const Exp<T>& x) {
    lm -= x.lv;
  }

  void multiply_in(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape == x.shape);
    eta += x.eta;
    lambda += x.lambda;
    lm += x.lm;
  }

  void divide_in(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape == x.shape);
    eta -= x.eta;
    lambda -= x.lambda;
    lm -= x.lm;
  }

  // Join operations
  //--------------------------------------------------------------------------

  void multiply_front(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    result.reset(clone());
    result.multiply_in_front(other);
  }

  void multiply_back(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    result.reset(clone());
    result.multiply_in_back(other);
  }

  void divide_front(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    result.reset(clone());
    result.divide_in_front(other);
  }

  void divide_back(const CanonicalGaussian& other, CanonicalGaussian& result) const {
    result.reset(clone());
    result.divide_in_back(other);
  }

  void multiply_dims(const CanonicalGaussian& other, const Dims& i, const Dims& j, CanonicalGaussian& result) const {
    result.reset(join(shape, other.shape(), i, j));
    const Impl& x = other.impl();
    Impl& impl = result.impl();
    Spans is = impl.spans(i);
    Spans js = impl.spans(j);
    sub(impl.eta, is) = eta;
    sub(impl.eta, js) += x.eta;
    sub(impl.lambda, is, is) = lambda;
    sub(impl.lambda, js, js) += x.lambda;
    impl.lm = lm + x.lm;
  }

  void divide_dims(const CanonicalGaussian& other, const Dims& i, const Dims& j, CanonicalGaussian& result) const {
    result.reset(join(shape, other.shape(), i, j));
    const Impl& x = other.impl();
    Impl& impl = result.impl();
    Spans is = impl.spans(i);
    Spans js = impl.spans(j);
    sub(impl.eta, is) = eta;
    sub(impl.eta, js) -= x.eta;
    sub(impl.lambda, is, is) = lambda;
    sub(impl.lambda, js, js) -= x.lambda;
    impl.lm = lm - x.lm;
  }

  void multiply_in_front(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape.has_prefix(x.shape));
    size_t n = x.size();
    eta.head(n) += x.eta;
    lambda.topLeftCorner(n, n) += x.lambda;
    lm += x.lm;
  }

  void multiply_in_back(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape.has_suffix(x.shape));
    size_t n = x.size();
    eta.tail(n) += x.eta;
    lambda.bottomRightCorner(n, n) += x.lambda;
    lm += x.lm;
  }

  void multiply_in_dims(const CanonicalGaussian& other, const Dims& dims) {
    const Impl& x = other.impl();
    assert(shape.has_select(dims, x.shape));
    Spans is = spans(dims);
    sub(eta, is) += x.eta;
    sub(lambda, is, is) += x.lambda;
    lm += x.lm;
  }

  void divide_in_front(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape.has_prefix(x.shape));
    size_t n = x.size();
    eta.head(n) -= x.eta;
    lambda.topLeftCorner(n, n) -= x.lambda;
    lm -= x.lm;
  }

  void divide_in_back(const CanonicalGaussian& other) {
    const Impl& x = other.impl();
    assert(shape.has_suffix(x.shape));
    size_t n = x.size();
    eta.tail(n) -= x.eta;
    lambda.bottomRightCorner(n, n) -= x.lambda;
    lm -= x.lm;
  }

  void divide_in_dims(const CanonicalGaussian& other, const Dims& dims) {
    const Impl& x = other.impl();
    assert(shape.has_select(dims, x.shape));
    Spans is = spans(dims);
    sub(eta, is) -= x.eta;
    sub(lambda, is, is) -= x.lambda;
    lm -= x.lm;
  }

  // Arithmetic operations
  //--------------------------------------------------------------------------

  void power(T alpha, CanonicalGaussian& result) const {
    result.reset(new Impl(shape, eta * alpha, lambda * alpha, lm * alpha));
  }

  void weighted_update(const CanonicalGaussian& other, T a, CanonicalGaussian& result) const {
    const Impl& x = other.impl();
    assert(shape == other.shape());
    T b = 1 - a;
    result.reset(new Impl(shape, b*eta + a*x.eta, b*lambda + a*x.lambda, b*lm + a*x.lm));
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

  Exp<T> maximum(RealValues<T>* values) const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "maximum");
    Vector<T> vec = chol.solve(eta);
    if (values) *values = vec;
    return Exp<T>(lm + eta.dot(vec) / T(2));
  }

  void marginal_front(unsigned n, CanonicalGaussian& result) const {
    reduce_front(n).marginal(result);
  }

  void marginal_back(unsigned n, CanonicalGaussian& result) const {
    reduce_back(n).marginal(result);
  }

  void marginal_dims(const Dims& dims, CanonicalGaussian& result) const {
    reduce(dims).marginal(result);
  }

  void maximum_front(unsigned n, CanonicalGaussian& result) const {
    reduce_front(n).maximum(result);
  }

  void maximum_back(unsigned n, CanonicalGaussian& result) const {
    reduce_back(n).maximum(result);
  }

  void maximum_dims(const Dims& dims, CanonicalGaussian& result) const {
    reduce(dims).maximum(result);
  }

  // Normalization
  //--------------------------------------------------------------------------

  void normalize() {
    lm -= marginal().lv;
  }

  void normalize(unsigned n) {
    // determine the block sizes
    size_t nx = shape.prefix_sum(n);
    size_t ny = size() - nx;

    // compute sol_xy = lam_xx^{-1} * lam_xy and sol_x = lam_xx^{-1} eta_x
    CholeskyType chol_xx;
    Matrix<T> sol_xy = lambda.topRightCorner(nx, ny);
    Vector<T> sol_x = eta.head(nx);
    chol_xx.compute(lambda.topLeftCorner(nx, nx));
    check(chol_xx, "normalize");
    if (nx) {
      chol_xx.solveInPlace(sol_xy);
      chol_xx.solveInPlace(sol_x);
    }

    // update the distribution
    eta.tail(ny).noalias() = sol_xy.transpose() * eta.head(nx);
    lambda.bottomRightCorner(ny, ny).noalias() = lambda.bottomLeftCorner(ny, nx) * sol_xy;
    lm = (logdet(chol_xx) - log_two_pi * nx - eta.head(nx).dot(sol_x)) / T(2);
  }

  // Restrictions
  //--------------------------------------------------------------------------

  void restrict_front(const RealValues<T>& values, CanonicalGaussian& result) const {
    reduce_front(values.size()).restrict(values, result);
  }

  void restrict_back(const RealValues<T>& values, CanonicalGaussian& result) const {
    reduce_back(values.size()).restrict(values, result);
  }

  void restrict_dims(const Dims& dims, const RealValues<T>& values, CanonicalGaussian& result) const {
    reduce(dims).restrict(values, result);
  }

  // Entropy and divergences
  //--------------------------------------------------------------------------

  T entropy() const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "entropy");
    return (size() * (log_two_pi + T(1)) - logdet(chol)) / T(2);
  }

  T kl_divergence(const CanonicalGaussian& other) const {
    const Impl& p = *this;
    const Impl& q = other.impl();
    assert(p.shape == q.shape);
    CholeskyType chol_p, chol_q;
    chol_p.compute(p.lambda);
    chol_q.compute(q.lambda);
    check(chol_p, "kl_divergence (p)");
    check(chol_q, "kl_divergence (q)");
    Vector<T> diff = chol_p.solve(p.eta) - chol_q.solve(q.eta);
    auto identity = Matrix<T>::Identity(p.size(), p.size());
    T trace = (q.lambda.array() * chol_p.solve(identity).array()).sum();
    T means = diff.transpose() * q.lambda * diff;
    T logdets = logdet(chol_p) - logdet(chol_q);
    return (trace + means + logdets - p.size()) / T(2);
  }

  T max_difference(const CanonicalGaussian& other) const {
    const Impl& x = other.impl();
    assert(shape == x.shape);
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
    MAT lam_xx, lam_xy, lam_yy;
    T lm;

    void collapse(CanonicalGaussian& result, bool adjust_lm) {
      // compute sol_yx = lam_yy^{-1} * lam_yx and sol_y = lam_yy^{-1} eta_y
      // using the Cholesky decomposition
      Eigen::LLT<Matrix<T>> chol_yy;
      Matrix<T> sol_yx = lam_xy.transpose();
      Vector<T> sol_y = eta_y;
      chol_yy.compute(lam_yy);
      check(chol_yy, "collapse");
      if (eta_y.size() != 0) {
        chol_yy.solveInPlace(sol_yx);
        chol_yy.solveInPlace(sol_y);
      }

      // compute the aggregate parameters
      Impl& impl = result.impl();
      impl.shape = std::move(shape);
      impl.eta = std::move(eta_x);
      impl.eta.noalias() -= sol_yx.transpose() * eta_y;
      impl.lambda = std::move(lam_xx);
      impl.lambda.noalias() -= lam_xy * sol_yx;
      impl.lm = lm + eta_y.dot(sol_y) / T(2);

      // adjust the log-multiplier if computing the marginal
      if (adjust_lm) {
        impl.lm += (log_two_pi * eta_y.size() - logdet(chol_yy)) / T(2);
      }
    }

    void marginal(CanonicalGaussian& result) {
      collapse(result, /*adjust_lm=*/true);
    }

    void maximum(CanonicalGaussian& result) {
      collapse(result, /*adjust_lm=*/false);
    }

    void restrict(const RealValues<T>& values, CanonicalGaussian& result) {
      const Vector<T>& vec = values.vec();
      Impl& impl = result.impl();
      impl.shape = std::move(shape);
      impl.eta = std::move(eta_x);
      impl.eta.noalias() -= lam_xy * vec;
      impl.lambda = std::move(lam_xx);
      impl.lm = lm + eta_y.dot(vec) - T(0.5) * vec.transpose() * lam_yy * vec;
    }
  };

  Reduce<BlockType, SegmentType> reduce_front(unsigned n) const {
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

  Reduce<BlockType, SegmentType> reduce_back(unsigned n) const {
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

  Reduce<Matrix<T>, Vector<T>> reduce(const Dims& i) const {
    Spans is = shape.spans(i);
    Spans js = shape.spans(~i, /*ignore_out_of_range=*/true);
    return {
      shape.select(i),
      sub(eta, is),
      sub(eta, js),
      sub(lambda, is, is),
      sub(lambda, is, js),
      sub(lambda, js, js),
      lm,
    };
  }

  // Conversion
  //--------------------------------------------------------------------------
  MomentGaussian<T> moment() const {
    CholeskyType chol;
    chol.compute(lambda);
    check(chol, "moment");
    Vector<T> mean = chol.solve(eta);
    T adjust = (size() * log_two_pi - logdet(chol) + mean.dot(eta)) / T(2);
    return {shape, mean, chol.solve(Matrix<T>::Identity(size(), size())), lm + adjust};
  }

  // Evaluation
  //--------------------------------------------------------------------------
  T log(const RealValues<T>& values) const {
    const Vector<T>& x = values.vec();
    return -T(0.5) * x.transpose() * lambda * x + eta.dot(x) + lm;
  }

}; // class Impl

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Exp<T> value)
  : Object(std::make_unique<Impl>(value)) {}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Shape shape, Exp<T> value)
  : Object(std::make_unique<Impl>(std::move(shape), value)) {}

template <typename T>
CanonicalGaussian<T>::CanonicalGaussian(Shape shape, Vector<T> eta, Matrix<T> lambda, T lv)
  : Object(std::make_unique<Impl>(std::move(shape), std::move(eta), std::move(lambda), lv)) {}

template <typename T>
unsigned CanonicalGaussian<T>::arity() const {
  return impl().shape.size();
}

template <typename T>
void CanonicalGaussian<T>::reset(Shape shape) {
  impl_ = std::make_unique<Impl>(std::move(shape), Exp<T>(0));
}

template <typename T>
void CanonicalGaussian<T>::reset(Impl* impl) {
  impl_.reset(impl);
}

template <typename T>
const Shape& CanonicalGaussian<T>::shape() const {
  return impl().shape;
}

template <typename T>
T CanonicalGaussian<T>::log_multiplier() const {
  return impl().lm;
}

template <typename T>
const Vector<T>& CanonicalGaussian<T>::inf_vector() const {
  return impl().eta;
}

template <typename T>
const Matrix<T>& CanonicalGaussian<T>::inf_matrix() const {
  return impl().lambda;
}

template <typename T>
Exp<T> CanonicalGaussian<T>::operator()(const RealValues<T>& values) const {
  return Exp<T>(log(values));
}

template <typename T>
T CanonicalGaussian<T>::log(const RealValues<T>& values) const {
  return impl().log(values);
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::operator*(const Exp<T>& x) const {
  CanonicalGaussian result;
  impl().multiply(x, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::operator*(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().multiply(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::operator*=(const Exp<T>& x) {
  impl().multiply_in(x);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::operator*=(const CanonicalGaussian& other) {
  impl().multiply_in(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::operator/(const Exp<T>& x) const {
  CanonicalGaussian result;
  impl().divide(x, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::divide_inverse(const Exp<T>& x) const {
  CanonicalGaussian result;
  impl().divide_inverse(x, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::operator/(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().divide(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::operator/=(const Exp<T>& x) {
  impl().divide_in(x);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::operator/=(const CanonicalGaussian& other) {
  impl().divide_in(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::multiply_front(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().multiply_front(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::multiply_back(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().multiply_back(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::multiply(const CanonicalGaussian& other, const Dims& i, const Dims& j) const {
  CanonicalGaussian result;
  impl().multiply_dims(other, i, j, result);
  return result;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::multiply_in_front(const CanonicalGaussian& other) {
  impl().multiply_in_front(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::multiply_in_back(const CanonicalGaussian& other) {
  impl().multiply_in_back(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::multiply_in(const CanonicalGaussian& other, const Dims& dims) {
  impl().multiply_in_dims(other, dims);
  return *this;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::divide_front(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().divide_front(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::divide_back(const CanonicalGaussian& other) const {
  CanonicalGaussian result;
  impl().divide_back(other, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::divide(const CanonicalGaussian& other, const Dims& i, const Dims& j) const {
  CanonicalGaussian result;
  impl().divide_dims(other, i, j, result);
  return result;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::divide_in_front(const CanonicalGaussian& other) {
  impl().divide_in_front(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::divide_in_back(const CanonicalGaussian& other) {
  impl().divide_in_back(other);
  return *this;
}

template <typename T>
CanonicalGaussian<T>& CanonicalGaussian<T>::divide_in(const CanonicalGaussian& other, const Dims& dims) {
  impl().divide_in_dims(other, dims);
  return *this;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::pow(T alpha) const {
  CanonicalGaussian result;
  impl().power(alpha, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::weighted_update(const CanonicalGaussian& other, T alpha) const {
  CanonicalGaussian result;
  impl().weighted_update(other, alpha, result);
  return result;
}

template <typename T>
Exp<T> CanonicalGaussian<T>::marginal() const {
  return impl().marginal();
}

template <typename T>
Exp<T> CanonicalGaussian<T>::maximum(RealValues<T>* values) const {
  return impl().maximum(values);
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::marginal_front(unsigned n) const {
  CanonicalGaussian result;
  impl().marginal_front(n, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::marginal_back(unsigned n) const {
  CanonicalGaussian result;
  impl().marginal_back(n, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::marginal_dims(const Dims& dims) const {
  CanonicalGaussian result;
  impl().marginal_dims(dims, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::maximum_front(unsigned n) const {
  CanonicalGaussian result;
  impl().maximum_front(n, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::maximum_back(unsigned n) const {
  CanonicalGaussian result;
  impl().maximum_back(n, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::maximum_dims(const Dims& dims) const {
  CanonicalGaussian result;
  impl().maximum_dims(dims, result);
  return result;
}

template <typename T>
void CanonicalGaussian<T>::normalize() {
  impl().normalize();
}

template <typename T>
void CanonicalGaussian<T>::normalize_head(unsigned nhead) {
  impl().normalize(nhead);
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::restrict_front(const RealValues<T>& values) const {
  CanonicalGaussian result;
  impl().restrict_front(values, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::restrict_back(const RealValues<T>& values) const {
  CanonicalGaussian result;
  impl().restrict_back(values, result);
  return result;
}

template <typename T>
CanonicalGaussian<T> CanonicalGaussian<T>::restrict_dims(const Dims& dims, const RealValues<T>& values) const {
  CanonicalGaussian result;
  impl().restrict_dims(dims, values, result);
  return result;
}

template <typename T>
T CanonicalGaussian<T>::entropy() const {
  return impl().entropy();
}

template <typename T>
T CanonicalGaussian<T>::kl_divergence(const CanonicalGaussian& other) const {
  return impl().kl_divergence(other);
}

template <typename T>
T CanonicalGaussian<T>::max_diff(const CanonicalGaussian& other) const {
  return impl().max_difference(other);
}

template <typename T>
MomentGaussian<T> CanonicalGaussian<T>::moment() const {
  return impl().moment();
}

template <typename T>
typename CanonicalGaussian<T>::Impl& CanonicalGaussian<T>::impl() {
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  return *static_cast<Impl*>(impl_.get());
}

template <typename T>
const typename CanonicalGaussian<T>::Impl& CanonicalGaussian<T>::impl() const {
  assert(impl_);
  return *static_cast<const Impl*>(impl_.get());
}

template <typename T>
const T CanonicalGaussian<T>::Impl::log_two_pi = std::log(boost::math::constants::two_pi<T>());

} // namespace libgm
