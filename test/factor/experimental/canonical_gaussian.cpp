#define BOOST_TEST_MODULE canonical_gaussian
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/canonical_gaussian.hpp>
#include <libgm/factor/experimental/moment_gaussian.hpp>

#include "predicates.hpp"
#include "../../math/eigen/helpers.hpp"

namespace libgm { namespace experimental {
  template class canonical_gaussian<double>;
  template class canonical_gaussian<float>;
  }
  template struct canonical_gaussian_param<>;
}

using namespace libgm;

using mgaussian = experimental::moment_gaussian<>;
using cgaussian = experimental::canonical_gaussian<>;
using param_type = canonical_gaussian_param<double>;
using vec_type = real_vector<>;
using mat_type = real_matrix<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  cgaussian a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(cg_properties(a, 0));

  cgaussian b(3);
  BOOST_CHECK(cg_properties(b, 3));

  cgaussian c(logd(2.0));
  BOOST_CHECK(cg_properties(c, 0));
  BOOST_CHECK_CLOSE(c.log_multiplier(), std::log(2.0), 1e-8);

  cgaussian e(2, logd(4.0));
  BOOST_CHECK(cg_properties(e, 2));
  BOOST_CHECK_CLOSE(e.log_multiplier(), std::log(4.0), 1e-8);

  param_type paramsf(3, 1.0);
  cgaussian f(std::move(paramsf));
  BOOST_CHECK(cg_properties(f, 3));
  BOOST_CHECK_EQUAL(f.log_multiplier(), 1.0);
  BOOST_CHECK_EQUAL(f.inf_vector(), vec_type::Zero(3));
  BOOST_CHECK_EQUAL(f.inf_matrix(), mat_type::Zero(3, 3));

  param_type paramsg(vec2(1, 3), mat22(2, 1, 1, 2), 5.0);
  cgaussian g(paramsg);
  BOOST_CHECK(cg_properties(g, 2));
  BOOST_CHECK_EQUAL(g.log_multiplier(), 5.0);
  BOOST_CHECK_EQUAL(g.inf_vector(), vec2(1, 3));
  BOOST_CHECK_EQUAL(g.inf_matrix(), mat22(2, 1, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  cgaussian f;
  f = logd(2.0);
  BOOST_CHECK(cg_properties(f, 0));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(2.0), 1e-8);

  f.reset(3);
  BOOST_CHECK(cg_properties(f, 3));

  f = logd(3.0);
  BOOST_CHECK(cg_properties(f, 0));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(3.0), 1e-8);

  mat_type cov = mat22(2,1,1,2);
  mat_type lam = mat22(2,-1,-1,2) / 3.0;
  mgaussian mg(vec2(2, 1), cov, mat21(3,4), 1.5);
  f = mg.canonical();
  BOOST_CHECK(cg_properties(f, 3));
  mat_type lam3 = mat33(2,-1,-2,-1,2,-5,-2,-5,26)/3.0;
  double lm = 1.5-std::log(two_pi<double>())-0.5*std::log(3)-0.5*2;
  BOOST_CHECK(cg_params(f, vec3(1,0,-3), lam3, lm));

  cgaussian g(3);
  f.reset(2);
  swap(f, g);
  BOOST_CHECK(cg_properties(f, 3));
  BOOST_CHECK(cg_properties(g, 2));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  cgaussian f(vec3(2, 1, 0), 2*mat_type::Identity(3, 3), 0.5);
  vec_type val = vec3(0.5, -2, 0);
  BOOST_CHECK_CLOSE(f.log(val), -0.5*8.5 + 2*0.5 - 1*2 + 0.5, 1e-8);
  BOOST_CHECK_CLOSE(f(val).lv, -0.5*8.5 + 2*0.5 - 1*2 + 0.5, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_operators) {
  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 2, 1, 1, 1, 2);
  cgaussian f(eta, lambda, 5);
  cgaussian g(vec1(1), mat11(3), 6);

  cgaussian h;
  h = g * f.tail(1);
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h,
                        vec3(1.2, 2, 0.5),
                        mat33(5, 1, 1, 1, 2, 1, 1, 1, 2), 11.0));

  h = f.dim(2) * g;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, 1.2),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, 5), 11.0));

  h.dims({2}) *= g;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, 2.2),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, 8), 17.0));

  h = f.tail(1) / g;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, -0.8),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, -1), -1.0));

  h.dims(2, 1) /= g;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, -1.8),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, -4), -7.0));

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h *= logd(1.0, log_tag());
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, eta, lambda, 8.0));

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h /= logd(1.0, log_tag());
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, eta, lambda, 6.0));

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, eta, lambda, 3.0));

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, -eta, -lambda, -3.0));

  h = pow(f, 2.0);
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, 2*eta, 2*lambda, 10.0));

  h = weighted_update(f, h, 0.3);
  BOOST_CHECK(cg_properties(h, 3));
  BOOST_CHECK(cg_params(h, 1.3*eta, 1.3*lambda, 1.3*5));
}

BOOST_AUTO_TEST_CASE(test_collapse) {
  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 2, 1, 1, 1, 2);
  cgaussian f(eta, lambda, 2.0);

  // test all marginal
  cgaussian h = f;
  BOOST_CHECK(h.normalizable());
  h.normalize();
  BOOST_CHECK_SMALL(h.sum().lv, 1e-8);

  // test block marginal
  h = f.marginal(0, 2);
  mat_type lamxy = lambda.inverse().eval().block(0, 0, 2, 2).inverse();
  vec_type etaxy = lamxy * (lambda.inverse() * eta).segment(0, 2);
  double cxy = 2.0 + 0.5*(std::log(pi<double>()) + 0.02);
  BOOST_CHECK(cg_properties(h, 2));
  BOOST_CHECK(cg_params(h, etaxy, lamxy, cxy));

  // test plain marginal
  h = f.marginal({2, 0});
  ivec ind = {2, 0};
  mat_type lamzx = submat(lambda.inverse().eval(), ind, ind).inverse();
  vec_type etazx = lamzx * subvec((lambda.inverse() * eta).eval(), ind);
  double czx = 2.0 + 0.5*(std::log(pi<double>()) + 0.125);
  BOOST_CHECK(cg_properties(h, 2));
  BOOST_CHECK(cg_params(h, etazx, lamzx, czx));

  // test maximum assignment
  vec_type vec;
  logd max = f.max(vec);
  vec_type mean = lambda.inverse() * eta;
  BOOST_CHECK(vec.isApprox(mean, 1e-8));
  BOOST_CHECK_CLOSE(f.log(mean), max.lv, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  cgaussian f(eta, lambda, 2.0);

  // test block restrict (z) = (1.5)
  cgaussian h = f.restrict(2, 1, vec1(1.5));
  BOOST_CHECK(cg_properties(h, 2));
  BOOST_CHECK(cg_params(h, vec2(2-1.5, 0.5-1.5), mat22(2, 1, 1, 3), 2.3-1.5*1.5*2));

  // test plain restrict (x, z) = (0.5, 1.5)
  h = f.restrict({0, 2}, vec2(0.5, 1.5));
  double c = -0.5 * vec2(0.5, 1.5).transpose() * mat22(2, 1, 1, 4) * vec2(0.5, 1.5);
  BOOST_CHECK(cg_properties(h, 1));
  BOOST_CHECK(cg_params(h, vec1(0.5-2.0), mat11(3), 2.0 + 1.3 + c));

  // test restrict-multiply
  param_type p = h.param();
  h = cgaussian(vec1(1), mat11(1.5), 2.0);
  h *= f.restrict({0, 2}, vec2(0.5, 1.5));
  BOOST_CHECK(cg_properties(h, 1));
  BOOST_CHECK(cg_params(h, p.eta + vec1(1), p.lambda + mat11(1.5), p.lm + 2.0));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mat_type cov = lambda.inverse();
  cgaussian p(eta, lambda, 2.0);

  double l2pi = std::log(two_pi<double>());
  double ent_xyz = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent_xyz / 2.0, 1e-5);

  double ent_x = std::log(cov.block(0, 0, 1, 1).determinant()) + l2pi + 1.0;
  BOOST_CHECK_CLOSE(p.entropy(0), ent_x / 2.0, 1e-5);

  cgaussian q(eta, lambda + mat_type::Identity(3, 3));
  BOOST_CHECK_GE(kl_divergence(p, q), 0.0);
  BOOST_CHECK_SMALL(kl_divergence(p, p), 1e-6);
}
