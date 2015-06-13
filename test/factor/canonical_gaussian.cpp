#define BOOST_TEST_MODULE canonical_gaussian
#include <boost/test/unit_test.hpp>

#include <libgm/factor/canonical_gaussian.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/moment_gaussian.hpp>

#include "predicates.hpp"
#include "../math/eigen/helpers.hpp"

namespace libgm {
  template class canonical_gaussian<double, variable>;
  template class canonical_gaussian<float, variable>;
  template class canonical_gaussian_param<double>;
  template class canonical_gaussian_param<float>;
}

using namespace libgm;

typedef canonical_gaussian_param<double> param_type;
typedef real_vector<double> vec_type;
typedef real_matrix<double> mat_type;

boost::test_tools::predicate_result
cg_properties(const cgaussian& f,
              const domain& vars) {
  std::size_t n = num_dimensions(vars);

  if (f.empty() && !vars.empty()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "The factor is empty [" << f << "]";
    return result;
  }
  if (f.arity() != vars.size()) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor arity ["
                     << f.arity() << " != " << vars.size() << "]";
    return result;
  }
  if (f.size() != n) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor size ["
                     << f.size() << " != " << n << "]";
    return result;
  }
  if (f.arguments() != vars) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Invalid factor domain ["
                     << f.arguments() << " != " << vars << "]";
    return result;
  }
  return true;
}

boost::test_tools::predicate_result
cg_params(const cgaussian& f,
          const vec_type& eta, const mat_type& lambda, double lm) {
  if (!f.inf_vector().isApprox(eta, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Information vectors not close: "
                     << f.inf_vector().transpose() << " vs "
                     << eta.transpose();
    return result;
  }
  if (!f.inf_matrix().isApprox(lambda, 1e-8)) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Information matrices not close: \n"
                     << f.inf_matrix() << "\n vs \n" << lambda;
    return result;
  }
  if (std::abs(f.log_multiplier() - lm) > 1e-8) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Log-multipliers not close: "
                     << f.log_multiplier() << " vs " << lm;
    return result;
  }
  return true;
}

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  variable x = u.new_continuous_variable("x", 2);
  variable y = u.new_continuous_variable("y", 1);

  cgaussian a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(cg_properties(a, {}));

  cgaussian b({x, y});
  BOOST_CHECK(cg_properties(b, {x, y}));

  cgaussian c(logd(2.0));
  BOOST_CHECK(cg_properties(c, {}));
  BOOST_CHECK_CLOSE(c.log_multiplier(), std::log(2.0), 1e-8);

  cgaussian e({x}, logd(4.0));
  BOOST_CHECK(cg_properties(e, {x}));
  BOOST_CHECK_CLOSE(e.log_multiplier(), std::log(4.0), 1e-8);

  param_type paramsf(3, 1.0);
  cgaussian f({x, y}, std::move(paramsf));
  BOOST_CHECK(cg_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(f.log_multiplier(), 1.0);
  BOOST_CHECK_EQUAL(f.inf_vector(), vec_type::Zero(3));
  BOOST_CHECK_EQUAL(f.inf_matrix(), mat_type::Zero(3, 3));

  param_type paramsg(vec2(1, 3), mat22(2, 1, 1, 2), 5.0);
  cgaussian g({x}, paramsg);
  BOOST_CHECK(cg_properties(g, {x}));
  BOOST_CHECK_EQUAL(g.log_multiplier(), 5.0);
  BOOST_CHECK_EQUAL(g.inf_vector(), vec2(1, 3));
  BOOST_CHECK_EQUAL(g.inf_matrix(), mat22(2, 1, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  variable x = u.new_continuous_variable("x", 2);
  variable y = u.new_continuous_variable("y", 1);

  cgaussian f;
  f = logd(2.0);
  BOOST_CHECK(cg_properties(f, {}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(2.0), 1e-8);

  f.reset({x, y});
  BOOST_CHECK(cg_properties(f, {x, y}));

  f = logd(3.0);
  BOOST_CHECK(cg_properties(f, {}));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(3.0), 1e-8);

  mat_type cov = mat22(2,1,1,2);
  mat_type lam = mat22(2,-1,-1,2) / 3.0;
  mgaussian mg({x}, {y}, vec2(2, 1), cov, mat21(3,4), 1.5);
  f = mg;
  BOOST_CHECK(cg_properties(f, {x, y}));
  mat_type lam3 = mat33(2,-1,-2,-1,2,-5,-2,-5,26)/3.0;
  double lm = 1.5-std::log(two_pi<double>())-0.5*std::log(3)-0.5*2;
  BOOST_CHECK(cg_params(f, vec3(1,0,-3), lam3, lm));

  cgaussian g({x, y});
  f.reset({x});
  swap(f, g);
  BOOST_CHECK(cg_properties(f, {x, y}));
  BOOST_CHECK(cg_properties(g, {x}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  variable x = u.new_continuous_variable("x", 2);
  variable y = u.new_continuous_variable("y", 1);

  cgaussian f({x, y}, vec3(2, 1, 0), 2*mat_type::Identity(3, 3), 0.5);
  vec_type vec = vec3(0.5, -2, 0);
  BOOST_CHECK_CLOSE(f.log(vec), -0.5*8.5 + 2*0.5 - 1*2 + 0.5, 1e-8);

  real_assignment<double> a;
  f.assignment(vec3(3, 2, 1), a);
  BOOST_CHECK_EQUAL(a[x], vec2(3, 2));
  BOOST_CHECK_EQUAL(a[y], vec1(1));

  variable v = u.new_continuous_variable("v", 2);
  variable w = u.new_continuous_variable("w", 1);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(cg_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  variable x = u.new_continuous_variable("x", 2);
  variable y = u.new_continuous_variable("y", 1);

  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 2, 1, 1, 1, 2);
  cgaussian f({x, y}, eta, lambda, 5);
  cgaussian g({y}, vec1(1), mat11(3), 6);

  cgaussian h;
  h = g * f;
  BOOST_CHECK(cg_properties(h, {y, x}));
  BOOST_CHECK(cg_params(h,
                        vec3(1.2, 2, 0.5),
                        mat33(5, 1, 1, 1, 2, 1, 1, 1, 2), 11.0));

  h = f * g;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, 1.2),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, 5), 11.0));

  h *= g;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, 2.2),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, 8), 17.0));

  h = f / g;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, -0.8),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, -1), -1.0));

  h /= g;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h,
                        vec3(2, 0.5, -1.8),
                        mat33(2, 1, 1, 1, 2, 1, 1, 1, -4), -7.0));

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h *= logd(1.0, log_tag());
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, eta, lambda, 8.0));

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, eta, lambda, 7.0));

  h /= logd(1.0, log_tag());
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, eta, lambda, 6.0));

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, eta, lambda, 3.0));

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, -eta, -lambda, -3.0));

  h = pow(f, 2.0);
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, 2*eta, 2*lambda, 10.0));

  h = weighted_update(f, h, 0.3);
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, 1.3*eta, 1.3*lambda, 1.3*5));
}

BOOST_AUTO_TEST_CASE(test_collapse) {
  universe u;
  variable x = u.new_continuous_variable("x", 1);
  variable y = u.new_continuous_variable("y", 1);
  variable z = u.new_continuous_variable("z", 1);

  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 2, 1, 1, 1, 2);
  cgaussian f({x, y, z}, eta, lambda, 2.0);


  // test all marginal
  cgaussian h = f;
  BOOST_CHECK(h.normalizable());
  h.normalize();
  BOOST_CHECK_SMALL(h.marginal().lv, 1e-8);

  // test block marginal
  h = f.marginal({x, y});
  matrix_index ind(0, 2);
  mat_type lamxy = submat(lambda.inverse().eval(), ind, ind).plain().inverse();
  vec_type etaxy = lamxy * subvec((lambda.inverse() * eta).eval(), ind).plain();
  double cxy = 2.0 + 0.5*(std::log(pi<double>()) + 0.02);
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, etaxy, lamxy, cxy));

  // test plain marginal
  h = f.marginal({z, x});
  ind = {2, 0};
  mat_type lamzx = submat(lambda.inverse().eval(), ind, ind).plain().inverse();
  vec_type etazx = lamzx * subvec((lambda.inverse() * eta).eval(), ind).plain();
  double czx = 2.0 + 0.5*(std::log(pi<double>()) + 0.125);
  BOOST_CHECK(cg_properties(h, {z, x}));
  BOOST_CHECK(cg_params(h, etazx, lamzx, czx));

  // test maximum assignment
  real_assignment<double> a;
  logd max = f.maximum(a);
  vec_type mean = lambda.inverse() * eta;
  BOOST_CHECK_CLOSE(a[x][0], mean[0], 1e-8);
  BOOST_CHECK_CLOSE(a[y][0], mean[1], 1e-8);
  BOOST_CHECK_CLOSE(a[z][0], mean[2], 1e-8);
  BOOST_CHECK_CLOSE(f.log(mean), max.lv, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  variable x = u.new_continuous_variable("x", 1);
  variable y = u.new_continuous_variable("y", 1);
  variable z = u.new_continuous_variable("z", 1);

  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  cgaussian f({x, y, z}, eta, lambda, 2.0);

  // test block restrict (z) = (1.5)
  real_assignment<double> a;
  a[z] = vec1(1.5);
  cgaussian h = f.restrict(a);
  BOOST_CHECK(cg_properties(h, {x, y}));
  BOOST_CHECK(cg_params(h, vec2(2-1.5, 0.5-1.5), mat22(2, 1, 1, 3), 2.3-1.5*1.5*2));

  // test plain restrict (x, z) = (0.5, 1.5)
  a[x] = vec1(0.5);
  h = f.restrict(a);
  double c = -0.5 * vec2(0.5, 1.5).transpose() * mat22(2, 1, 1, 4) * vec2(0.5, 1.5);
  BOOST_CHECK(cg_properties(h, {y}));
  BOOST_CHECK(cg_params(h, vec1(0.5-2.0), mat11(3), 2.0 + 1.3 + c));

  // test restrict-multiply
  param_type p = h.param();
  h = cgaussian({y}, vec1(1), mat11(1.5), 2.0);
  f.restrict_multiply(a, h);
  BOOST_CHECK(cg_properties(h, {y}));
  BOOST_CHECK(cg_params(h, p.eta + vec1(1), p.lambda + mat11(1.5), p.lm + 2.0));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  universe u;
  variable x = u.new_continuous_variable("x", 1);
  variable y = u.new_continuous_variable("y", 1);
  variable z = u.new_continuous_variable("z", 1);

  vec_type eta = vec3(2, 0.5, 0.2);
  mat_type lambda = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mat_type cov = lambda.inverse();
  cgaussian p({x, y, z}, eta, lambda, 2.0);

  double l2pi = std::log(two_pi<double>());
  double ent_xyz = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent_xyz / 2.0, 1e-5);

  double ent_x = std::log(cov.block(0, 0, 1, 1).determinant()) + l2pi + 1.0;
  BOOST_CHECK_CLOSE(p.entropy({x}), ent_x / 2.0, 1e-5);

  cgaussian q({x, y, z}, eta, lambda + mat_type::Identity(3, 3));
  BOOST_CHECK_GE(kl_divergence(p, q), 0.0);
  BOOST_CHECK_SMALL(kl_divergence(p, p), 1e-6);
}

