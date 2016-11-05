#define BOOST_TEST_MODULE moment_gaussian
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/moment_gaussian.hpp>
#include <libgm/factor/experimental/canonical_gaussian.hpp>

#include "predicates.hpp"
#include "../../math/eigen/helpers.hpp"

namespace libgm { namespace experimental {
    template class moment_gaussian<double>;
    template class moment_gaussian<float>;
  }
  template struct moment_gaussian_param<>;
}

using namespace libgm;

using mgaussian = experimental::moment_gaussian<>;
using cgaussian = experimental::canonical_gaussian<>;
using param_type = moment_gaussian_param<double>;
using vec_type = real_vector<>;
using mat_type = real_matrix<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  mgaussian a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(mg_properties(a, 0));

  mgaussian b(3);
  BOOST_CHECK(mg_properties(b, 3));

  mgaussian c(3, 1);
  BOOST_CHECK(mg_properties(c, 3, 1));

  mgaussian d(logd(2.0));
  BOOST_CHECK(mg_properties(d, 0));
  BOOST_CHECK_CLOSE(d.log_multiplier(), std::log(2.0), 1e-8);

  param_type param(3, 0);
  param.mean = vec3(1, 2, 3);
  param.cov  = mat_type::Identity(3, 3);
  param.lm   = 2.0;
  mgaussian e(param);
  BOOST_CHECK(mg_properties(e, 3));
  BOOST_CHECK(mg_params(e, vec3(1,2,3), mat_type::Identity(3, 3), mat_type(3, 0), 2.0));
  mgaussian em(std::move(param));
  BOOST_CHECK(mg_properties(em, 3));
  BOOST_CHECK(mg_params(em, vec3(1,2,3), mat_type::Identity(3, 3), mat_type(3, 0), 2.0));

  param_type param1 = e.param();
  param1.coef = vec3(0.5, 1, 2);
  mgaussian f(param1);
  BOOST_CHECK(mg_properties(f, 3, 1));
  BOOST_CHECK(mg_params(f, vec3(1,2,3), mat_type::Identity(3, 3), vec3(0.5,1,2), 2.0));
  mgaussian fm(std::move(param1));
  BOOST_CHECK(mg_properties(fm, 3, 1));
  BOOST_CHECK(mg_params(fm, vec3(1,2,3), mat_type::Identity(3, 3), vec3(0.5,1,2), 2.0));

  mgaussian g(vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), 1.5);
  BOOST_CHECK(mg_properties(g, 3));
  BOOST_CHECK(mg_params(g, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), mat_type(3, 0), 1.5));

  mgaussian h(vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), vec3(4,5,6), 1.5);
  BOOST_CHECK(mg_properties(h, 3, 1));
  BOOST_CHECK(mg_params(h, vec3(3,2,1), mat33(2,1,1,1,2,1,1,1,3), vec3(4,5,6), 1.5));
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  mgaussian f;
  f = logd(2.0);
  BOOST_CHECK(mg_properties(f, 0));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(2.0), 1e-8);

  f.reset(3);
  BOOST_CHECK(mg_properties(f, 3));

  f = logd(3.0);
  BOOST_CHECK(mg_properties(f, 0));
  BOOST_CHECK_CLOSE(f.log_multiplier(), std::log(3.0), 1e-8);

  cgaussian cg(vec2(1,2), mat22(2,0,0,2), 1.5);
  f = cg.moment();
  double lm = 1.5 + std::log(two_pi<double>()) - 0.5*std::log(4) + 0.5*2.5;
  BOOST_CHECK(mg_properties(f, 2));
  BOOST_CHECK(mg_params(f, vec2(0.5,1), mat22(0.5,0,0,0.5), mat_type(2,0), lm));
  BOOST_CHECK_CLOSE(f.max().lv, cg.max().lv, 1e-8);
  BOOST_CHECK_CLOSE(f.sum().lv, cg.sum().lv, 1e-8);
  BOOST_CHECK_CLOSE(f(vec2(1,-3)).lv, cg(vec2(1,-3)).lv, 1e-8);

  mgaussian g(3);
  swap(f, g);
  BOOST_CHECK(mg_properties(f, 3));
  BOOST_CHECK(mg_properties(g, 2));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  mgaussian f(vec3(2, 1, 0), 2*mat_type::Identity(3, 3), 0.5);
  vec_type val = vec3(0.5, -2, 0);
  double lv = -0.25*(1.5*1.5+3*3)-1.5*log(two_pi<double>())-0.5*log(8)+0.5;
  BOOST_CHECK_CLOSE(f.log(val), lv, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_multiplication) {
  // small test
  mgaussian f(vec1(1), mat11(2), mat12(0.5, 3), 1.2); // x | y
  mgaussian g(vec2(2, 1), mat22(2,1,1,2), 0.3);       // y

  mgaussian h = f.tail() * g.head();
  BOOST_CHECK(mg_properties(h, 3));
  BOOST_CHECK(mg_params(h,
                        vec3(1+1+3,2,1),
                        mat33(2+0.5+3+18,1+3,0.5+6,1+3,2,1,0.5+6,1,2),
                        mat_type(3,0),
                        1.5));

  mgaussian h2 = g.head() * f.tail();
  param_type p2 = h.reorder(uint_vector{1, 2, 0}).param();
  BOOST_CHECK(mg_properties(h2, 3));
  BOOST_CHECK(mg_params(h2, p2.mean, p2.cov, p2.coef, p2.lm));

  // multiplication by a constant
  mgaussian h3 = h2 * logd(2.0, log_tag());
  BOOST_CHECK(mg_properties(h3, 3));
  BOOST_CHECK(mg_params(h3, p2.mean, p2.cov, p2.coef, p2.lm+2.0));

  mgaussian h4 = logd(2.0, log_tag()) * h2;
  BOOST_CHECK(mg_properties(h4, 3));
  BOOST_CHECK(mg_params(h4, p2.mean, p2.cov, p2.coef, p2.lm+2.0));
  h4 *= logd(3.0, log_tag());
  BOOST_CHECK_CLOSE(h4.log_multiplier(), p2.lm+5, 1e-8);

  // division by a constant
  mgaussian h5 = h2 / logd(2.0, log_tag());
  BOOST_CHECK(mg_properties(h5, 3));
  BOOST_CHECK(mg_params(h5, p2.mean, p2.cov, p2.coef, p2.lm-2.0));
  h5 /= logd(3.0, log_tag());
  BOOST_CHECK_CLOSE(h5.log_multiplier(), p2.lm-5, 1e-8);

  // large test (1, 2, 1)
  f = mgaussian(vec3(3,2,1), mat33(4,2,2,2,3,2,2,2,2.5), 1.5); // x, y
  g = mgaussian(vec1(0.5), mat11(0.8), mat12(0.1,0.2), 1.7);   // z | y
  h = f.head(1, 2) * g.tail();
  param_type p = (f.canonical().tail(2) * g.canonical().tail(2)).moment().param();
  BOOST_CHECK(mg_properties(h, 4));
  BOOST_CHECK(mg_params(h, p.mean, p.cov, mat_type(4,0), p.lm));
}

BOOST_AUTO_TEST_CASE(test_collapse) {
  vec_type mean = vec3(2, 0.5, 0.2);
  mat_type cov = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mgaussian f(mean, cov, 2.0); // x, y, z

  // test all marginal
  mgaussian h = f;
  BOOST_CHECK(h.normalizable());
  h.normalize();
  BOOST_CHECK_SMALL(h.sum().lv, 1e-8);

  // test block marginal
  h = f.marginal(0, 2);
  vec_type meanxy = mean.segment(0, 2);
  mat_type covxy = cov.block(0, 0, 2, 2);
  BOOST_CHECK(mg_properties(h, 2));
  BOOST_CHECK(mg_params(h, meanxy, covxy, mat_type(2,0), f.log_multiplier()));
  BOOST_CHECK_CLOSE(h.sum().lv, 2.0, 1e-8);

  // test subset marginal
  h = f.marginal({2, 0});
  ivec ind = {2, 0};
  vec_type meanzx = subvec(mean, ind);
  mat_type covzx = submat(cov, ind, ind);
  BOOST_CHECK(mg_properties(h, 2));
  BOOST_CHECK(mg_params(h, meanzx, covzx, mat_type(2,0), f.log_multiplier()));
  BOOST_CHECK_CLOSE(h.sum().lv, 2.0, 1e-8);

  // test conditional marginal
  mgaussian g(vec2(3,2), mat22(2,1,1,2), mat21(4,5), 1.2); // y, x, | z
  h = g.marginal(1);
  BOOST_CHECK(mg_properties(h, 1, 1));
  BOOST_CHECK(mg_params(h, vec1(2), mat11(2), mat11(5), 1.2));

  // test maximum assignment
  vec_type max_vec;
  logd max = f.max(max_vec);
  BOOST_CHECK_CLOSE(max_vec[0], mean[0], 1e-8);
  BOOST_CHECK_CLOSE(max_vec[1], mean[1], 1e-8);
  BOOST_CHECK_CLOSE(max_vec[2], mean[2], 1e-8);
  BOOST_CHECK_CLOSE(f.log(mean), max.lv, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  mgaussian f(vec2(3,4), mat22(2,1,1,2), mat22(4,5,2,-1), 2.0); // x, y | z, w

  // restrict all tail and some head
  mgaussian g = f.restrict_tail(vec2(2, 3)).restrict_head(1, 1, vec1(1));
  param_type p = f.canonical().restrict(1, 3, vec3(1, 2, 3)).moment().param();
  BOOST_CHECK(mg_properties(g, 1));
  BOOST_CHECK(mg_params(g, p.mean, p.cov, p.coef, p.lm));

  // restrict some tail and no head
  g = f.restrict_tail(1, 1, vec1(2));
  BOOST_CHECK(mg_properties(g, 2, 1));
  BOOST_CHECK(mg_params(g, vec2(3+5*2,4-1*2), mat22(2,1,1,2), mat21(4,2), 2.0));
}


BOOST_AUTO_TEST_CASE(test_sample) {
  std::mt19937 rng1;
  std::mt19937 rng2;

  // test marginal sample
  mgaussian f(vec2(3,4), mat22(2,1,1,2)); // x, y
  auto fd = f.distribution();
  for (std::size_t i = 0; i < 20; ++i) {
    vec_type sample = fd(rng1);
    BOOST_CHECK_EQUAL(sample.size(), 2);
    BOOST_CHECK_SMALL((sample - f.sample(rng2)).cwiseAbs().sum(), 1e-8);
  }

  // test conditional sample
  mgaussian g(vec2(3,4), mat22(2,1,1,2), mat21(2,-1), 2.0); // x, y | z
  auto gd = g.distribution();
  for (double zv = -1; zv < 1; zv += 0.2) {
    vec_type tail = vec_type::Constant(1, zv);
    for (std::size_t i = 0; i < 20; ++i) {
      vec_type sample = gd(rng1, tail);
      BOOST_CHECK_EQUAL(sample.size(), 2);
      BOOST_CHECK_SMALL((sample - g.restrict_tail(tail).sample(rng2))
                        .cwiseAbs().sum(), 1e-8);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  vec_type mean = vec3(2, 0.5, 0.2);
  mat_type cov = mat33(2, 1, 1, 1, 3, 1, 1, 1, 4);
  mgaussian p(mean, cov, 2.0); // x, y, z

  double l2pi = std::log(two_pi<double>());
  double ent_xyz = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent_xyz / 2.0, 1e-5);

  double ent_x = std::log(cov.block(0, 0, 1, 1).determinant()) + l2pi + 1.0;
  BOOST_CHECK_CLOSE(p.entropy(0), ent_x / 2.0, 1e-5);

  mgaussian q(mean, cov + mat_type::Identity(3, 3));
  BOOST_CHECK_GE(kl_divergence(p, q), 0.0);
  BOOST_CHECK_SMALL(kl_divergence(p, p), 1e-6);
}
