#define BOOST_TEST_MODULE canonical_gaussian
#include <boost/test/unit_test.hpp>

#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/impl/canonical_gaussian.hpp>

#include <algorithm>
#include <numeric>
#include "predicates.hpp"

#include <boost/math/constants/constants.hpp>

using namespace libgm;

using CGaussian = CanonicalGaussian<double>;
using Vec = Vector<double>;
using Mat = Matrix<double>;

namespace {

Vec vec(std::initializer_list<double> xs) {
  Vec out(xs.size());
  std::copy(xs.begin(), xs.end(), out.data());
  return out;
}

Mat mat(size_t rows, size_t cols, std::initializer_list<double> xs) {
  Mat out(rows, cols);
  std::copy(xs.begin(), xs.end(), out.data());
  return out;
}

Mat submatrix02(const Mat& m) {
  Mat out(2, 2);
  out << m(0, 0), m(0, 2),
         m(2, 0), m(2, 2);
  return out;
}

Vec subvector02(const Vec& v) {
  Vec out(2);
  out << v(0), v(2);
  return out;
}

Dims make_dims(std::initializer_list<size_t> idx) {
  Dims d;
  for (size_t i : idx) {
    d.set(i);
  }
  return d;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_constructors) {
  CGaussian a(Exp<double>(0.0));
  BOOST_CHECK(cg_properties(a, Shape{}));

  CGaussian b(Shape(3, 1));
  BOOST_CHECK(cg_properties(b, Shape(3, 1)));

  CGaussian c(Exp<double>(2.0));
  BOOST_CHECK(cg_properties(c, Shape{}));
  BOOST_CHECK_CLOSE(c.log_multiplier(), 2.0, 1e-8);

  CGaussian d(Shape(2, 1), Exp<double>(4.0));
  BOOST_CHECK(cg_properties(d, Shape(2, 1)));
  BOOST_CHECK_CLOSE(d.log_multiplier(), 4.0, 1e-8);

  CGaussian e(Shape(2, 1), vec({1, 3}), mat(2, 2, {2, 1, 1, 2}), 5.0);
  BOOST_CHECK(cg_properties(e, Shape(2, 1)));
  BOOST_CHECK(cg_params(e, vec({1, 3}), mat(2, 2, {2, 1, 1, 2}), 5.0));
}

BOOST_AUTO_TEST_CASE(test_assignment_swap_and_conversion) {
  CGaussian f;
  f = CGaussian(Exp<double>(2.0));
  BOOST_CHECK(cg_properties(f, Shape{}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), 2.0, 1e-8);

  f.reset(Shape(3, 1));
  BOOST_CHECK(cg_properties(f, Shape(3, 1)));

  f = CGaussian(Exp<double>(3.0));
  BOOST_CHECK(cg_properties(f, Shape{}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), 3.0, 1e-8);

  CGaussian g(Shape(2, 1));
  f.reset(Shape(3, 1));
  swap(f, g);
  BOOST_CHECK(cg_properties(f, Shape(2, 1)));
  BOOST_CHECK(cg_properties(g, Shape(3, 1)));
}

BOOST_AUTO_TEST_CASE(test_indexing) {
  CGaussian f(
    Shape(3, 1),
    vec({2, 1, 0}),
    2 * Mat::Identity(3, 3),
    0.5
  );
  Vec val = vec({0.5, -2, 0});
  BOOST_CHECK_CLOSE(f.log(val), -0.5 * 8.5 + 2 * 0.5 - 1 * 2 + 0.5, 1e-8);
  BOOST_CHECK_CLOSE(f(val).lv, -0.5 * 8.5 + 2 * 0.5 - 1 * 2 + 0.5, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_elementwise) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 5.0);
  CGaussian g(Shape(3, 1), vec({1, -2, 0.5}), mat(3, 3, {3, 0, 0, 0, 4, 0, 0, 0, 5}), 6.0);

  CGaussian h = f * g;
  BOOST_CHECK(cg_properties(h, Shape(3, 1)));
  BOOST_CHECK(cg_params(h, eta + g.inf_vector(), lambda + g.inf_matrix(), 11.0));

  h = f / g;
  BOOST_CHECK(cg_properties(h, Shape(3, 1)));
  BOOST_CHECK(cg_params(h, eta - g.inf_vector(), lambda - g.inf_matrix(), -1.0));

  h = f;
  h *= g;
  h /= g;
  BOOST_CHECK_SMALL(h.max_diff(f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_scalar) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 5.0);
  CGaussian h;

  h = f * Exp<double>(2.0);
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h *= Exp<double>(1.0);
  BOOST_CHECK(cg_params(h, eta, lambda, 8.0));

  h = Exp<double>(2.0) * f;
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h /= Exp<double>(1.0);
  BOOST_CHECK(cg_params(h, eta, lambda, 6.0));

  h = f / Exp<double>(2.0);
  BOOST_CHECK(cg_params(h, eta, lambda, 3.0));

  h = Exp<double>(2.0) / f;
  BOOST_CHECK(cg_params(h, -eta, -lambda, -3.0));
}

BOOST_AUTO_TEST_CASE(test_join_front_back) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 5.0);
  CGaussian g(Shape(1, 1), vec({1}), mat(1, 1, {3}), 6.0);

  CGaussian h = f.multiply_front(g);
  BOOST_CHECK(cg_params(h, vec({3, 0.5, 0.2}), mat(3, 3, {5, 1, 1, 1, 2, 1, 1, 1, 2}), 11.0));

  h = f.multiply_back(g);
  BOOST_CHECK(cg_params(h, vec({2, 0.5, 1.2}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 5}), 11.0));

  h = f.divide_front(g);
  BOOST_CHECK(cg_params(h, vec({1, 0.5, 0.2}), mat(3, 3, {-1, 1, 1, 1, 2, 1, 1, 1, 2}), -1.0));

  h = f.divide_back(g);
  BOOST_CHECK(cg_params(h, vec({2, 0.5, -0.8}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, -1}), -1.0));

  CGaussian in = f;
  in.multiply_in_front(g);
  BOOST_CHECK_SMALL(in.max_diff(f.multiply_front(g)), 1e-8);
  in = f;
  in.multiply_in_back(g);
  BOOST_CHECK_SMALL(in.max_diff(f.multiply_back(g)), 1e-8);
  in = f;
  in.divide_in_front(g);
  BOOST_CHECK_SMALL(in.max_diff(f.divide_front(g)), 1e-8);
  in = f;
  in.divide_in_back(g);
  BOOST_CHECK_SMALL(in.max_diff(f.divide_back(g)), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_join_dims) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 5.0);
  CGaussian g(Shape(1, 1), vec({1}), mat(1, 1, {3}), 6.0);
  CGaussian h;

  h = multiply(f, g, make_dims({0, 1, 2}), make_dims({2}));
  BOOST_CHECK(cg_params(h, vec({2, 0.5, 1.2}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 5}), 11.0));

  h.multiply_in(g, make_dims({2}));
  BOOST_CHECK(cg_params(h, vec({2, 0.5, 2.2}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 8}), 17.0));

  h = divide(f, g, make_dims({0, 1, 2}), make_dims({2}));
  BOOST_CHECK(cg_params(h, vec({2, 0.5, -0.8}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, -1}), -1.0));

  h.divide_in(g, make_dims({2}));
  BOOST_CHECK(cg_params(h, vec({2, 0.5, -1.8}), mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, -4}), -7.0));
}

BOOST_AUTO_TEST_CASE(test_arithmetic) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 5.0);

  CGaussian hpow = pow(f, 2.0);
  BOOST_CHECK(cg_params(hpow, 2 * eta, 2 * lambda, 10.0));

  CGaussian hwu = weighted_update(f, hpow, 0.3);
  BOOST_CHECK(cg_params(hwu, 1.3 * eta, 1.3 * lambda, 6.5));
}

BOOST_AUTO_TEST_CASE(test_marginal_and_maximum) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 2, 1, 1, 1, 2});
  CGaussian f(Shape(3, 1), eta, lambda, 2.0);

  CGaussian h = f;
  h.normalize();
  BOOST_CHECK_CLOSE(h.marginal().lv, 0.0, 1e-8);

  h = f.marginal_front(2);
  Mat sigma = lambda.inverse();
  Mat sigma_xy = sigma.topLeftCorner(2, 2);
  Mat lamxy = sigma_xy.inverse();
  Vec etaxy = lamxy * (sigma * eta).head(2);
  double cxy = 2.0 + 0.5 * (std::log(boost::math::constants::pi<double>()) + 0.02);
  BOOST_CHECK(cg_properties(h, Shape(2, 1)));
  BOOST_CHECK(cg_params(h, etaxy, lamxy, cxy));

  Vec argmax;
  Exp<double> mx = f.maximum(&argmax);
  Vec mean = lambda.inverse() * eta;
  BOOST_CHECK(argmax.isApprox(mean, 1e-8));
  BOOST_CHECK_CLOSE(f.log(mean), mx.lv, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 3, 1, 1, 1, 4});
  CGaussian f(Shape(3, 1), eta, lambda, 2.0);

  // Restrict z (dim {2}) = 1.5.
  CGaussian h = f.restrict_dims(make_dims({2}), vec({1.5}));
  BOOST_CHECK(cg_properties(h, Shape(2, 1)));
  BOOST_CHECK(cg_params(h, vec({0.5, -1.0}), mat(2, 2, {2, 1, 1, 3}), 2.3 - 1.5 * 1.5 * 2));

  // Restrict x,z (dims {0,2}) = (0.5, 1.5).
  h = f.restrict_dims(make_dims({0, 2}), vec({0.5, 1.5}));
  double c = -0.5 * vec({0.5, 1.5}).transpose() * mat(2, 2, {2, 1, 1, 4}) * vec({0.5, 1.5});
  BOOST_CHECK(cg_properties(h, Shape(1, 1)));
  BOOST_CHECK(cg_params(h, vec({-1.5}), mat(1, 1, {3}), 3.3 + c));

  Vec peta = h.inf_vector();
  Mat plam = h.inf_matrix();
  double plm = h.log_multiplier();
  h = CGaussian(Shape(1, 1), vec({1}), mat(1, 1, {1.5}), 2.0);
  h *= f.restrict_dims(make_dims({0, 2}), vec({0.5, 1.5}));
  BOOST_CHECK(cg_properties(h, Shape(1, 1)));
  BOOST_CHECK(cg_params(h, peta + vec({1}), plam + mat(1, 1, {1.5}), plm + 2.0));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  Vec eta = vec({2, 0.5, 0.2});
  Mat lambda = mat(3, 3, {2, 1, 1, 1, 3, 1, 1, 1, 4});
  Mat cov = lambda.inverse();
  CGaussian p(Shape(3, 1), eta, lambda, 2.0);

  double l2pi = std::log(boost::math::constants::two_pi<double>());
  double ent_xyz = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent_xyz / 2.0, 1e-5);

  CGaussian q(Shape(3, 1), eta + vec({0.1, -0.2, 0.15}), lambda);
  BOOST_CHECK_GT(p.kl_divergence(q), 0.0); // FIXME: check if this is correct
  BOOST_CHECK_SMALL(p.kl_divergence(p), 1e-6);
}

BOOST_AUTO_TEST_CASE(test_non_unit_shape_behavior) {
  Shape s = {2, 1};
  CGaussian f(s, vec({1.0, 2.0, 3.0}), Mat::Identity(3, 3), 0.5);
  BOOST_CHECK(cg_properties(f, s));

  CGaussian g(Shape{1}, vec({0.4}), mat(1, 1, {2.0}), 1.2);
  CGaussian h = f.multiply_back(g);
  BOOST_CHECK(cg_properties(h, s));
  BOOST_CHECK_CLOSE(h.inf_vector()(0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(h.inf_vector()(1), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(h.inf_vector()(2), 3.4, 1e-8);
  BOOST_CHECK_CLOSE(h.inf_matrix()(2, 2), 3.0, 1e-8);

  CGaussian m = f.maximum_back(1);
  BOOST_CHECK(cg_properties(m, Shape{1}));

  CGaussian mf = f.maximum_front(1);
  BOOST_CHECK(cg_properties(mf, Shape{2}));
}
