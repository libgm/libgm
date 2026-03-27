#define BOOST_TEST_MODULE moment_gaussian
#include <boost/test/unit_test.hpp>

#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/factor/impl/moment_gaussian.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/impl/canonical_gaussian.hpp>

#include "predicates.hpp"

#include <boost/math/constants/constants.hpp>

using namespace libgm;

using MGaussian = MomentGaussian<double>;
using CGaussian = CanonicalGaussian<double>;
using Vec = Vector<double>;
using Mat = Matrix<double>;

BOOST_AUTO_TEST_CASE(test_constructors_and_accessors) {
  const Vec mean2{{3.0, 2.0}};
  const Mat cov2{{2.0, 1.0}, {1.0, 2.0}};
  const Mat coef21{{4.0}, {5.0}};
  const Mat cov_id3 = Mat::Identity(3, 3);
  const Mat empty_coef2(2, 0);

  MGaussian a(Exp<double>(2.0));
  BOOST_CHECK(mg_properties(a, Shape{}));
  BOOST_CHECK_CLOSE(a.log_multiplier(), 2.0, 1e-8);

  MGaussian b(Shape{2, 1});
  BOOST_CHECK(mg_properties(b, Shape({2, 1})));
  BOOST_CHECK(b.mean().isApprox(Vec::Zero(3), 1e-12));
  BOOST_CHECK(b.covariance().isApprox(cov_id3, 1e-12));
  BOOST_CHECK_CLOSE(b.log_multiplier(), 0.0, 1e-8);

  MGaussian c(Shape{2}, mean2, cov2, 1.5);
  BOOST_CHECK(mg_properties(c, Shape({2})));
  BOOST_CHECK(mg_params(c, mean2, cov2, empty_coef2, 1.5));

  MGaussian d(Shape{2, 1},mean2,cov2,coef21, 1.2);
  BOOST_CHECK(mg_properties(d, Shape({2, 1}), 1));
  BOOST_CHECK(mg_params(d, mean2, cov2, coef21, 1.2));
}

BOOST_AUTO_TEST_CASE(test_indexing_and_scalar_ops) {
  MGaussian f(Shape{3}, Vec{{2.0, 1.0, 0.0}}, 2.0 * Mat::Identity(3, 3), 0.5);

  Vec val{{0.5, -2.0, 0.0}};
  const double l2pi = std::log(boost::math::constants::two_pi<double>());
  const double expected = -0.25 * (1.5 * 1.5 + 3.0 * 3.0) - 1.5 * l2pi - 0.5 * std::log(8.0) + 0.5;
  BOOST_CHECK_CLOSE(f.log(val), expected, 1e-8);
  BOOST_CHECK_CLOSE(f(val).lv, expected, 1e-8);

  MGaussian g = f * Exp<double>(2.0);
  BOOST_CHECK_CLOSE(g.log_multiplier(), f.log_multiplier() + 2.0, 1e-8);

  g *= Exp<double>(3.0);
  BOOST_CHECK_CLOSE(g.log_multiplier(), f.log_multiplier() + 5.0, 1e-8);

  MGaussian h = g / Exp<double>(1.0);
  BOOST_CHECK_CLOSE(h.log_multiplier(), f.log_multiplier() + 4.0, 1e-8);

  h /= Exp<double>(4.0);
  BOOST_CHECK_CLOSE(h.log_multiplier(), f.log_multiplier(), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_aggregates_and_normalize) {
  Vec mean{{2.0, 0.5, 0.2}};
  Mat cov{{2.0, 1.0, 1.0}, {1.0, 3.0, 1.0}, {1.0, 1.0, 4.0}};
  MGaussian f(Shape{1, 1, 1}, mean, cov, 2.0);

  MGaussian n = f;
  n.normalize();
  BOOST_CHECK(mg_properties(n, Shape({1, 1, 1})));
  BOOST_CHECK_CLOSE(n.marginal().lv, 0.0, 1e-8);

  MGaussian mfront = f.marginal_front(2);
  BOOST_CHECK(mg_properties(mfront, Shape({1, 1})));
  BOOST_CHECK(mg_params(mfront,
                        Vec{{2.0, 0.5}},
                        Mat{{2.0, 1.0}, {1.0, 3.0}},
                        Mat(2, 0),
                        2.0));

  MGaussian mback = f.marginal_back(2);
  BOOST_CHECK(mg_properties(mback, Shape({1, 1})));
  BOOST_CHECK(mg_params(mback,
                        Vec{{0.5, 0.2}},
                        Mat{{3.0, 1.0}, {1.0, 4.0}},
                        Mat(2, 0),
                        2.0));

  MGaussian mdims = f.marginal_dims(make_dims({2, 0}));
  BOOST_CHECK(mg_properties(mdims, Shape({1, 1})));
  BOOST_CHECK(mg_params(mdims,
                        Vec{{2.0, 0.2}},
                        Mat{{2.0, 1.0}, {1.0, 4.0}},
                        Mat(2, 0),
                        2.0));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  MGaussian f(Shape{2, 1},
              Vec{{3.0, 4.0, 5.0}},
              Mat{{2.0, 1.0, 0.5}, {1.0, 2.5, -0.3}, {0.5, -0.3, 1.7}},
              2.0);

  Vec front_vals{{1.0, 2.0}};
  Vec back_vals{{0.5}};
  Dims front_dims = make_dims({0});
  Dims back_dims = make_dims({1});

  MGaussian rf = f.restrict_front(front_vals);
  MGaussian rf_ref = f.canonical().restrict_front(front_vals).moment();
  BOOST_CHECK(mg_properties(rf, Shape({1})));
  BOOST_CHECK(mg_params(rf, rf_ref));

  MGaussian rb = f.restrict_back(back_vals);
  MGaussian rb_ref = f.canonical().restrict_back(back_vals).moment();
  BOOST_CHECK(mg_properties(rb, Shape({2})));
  BOOST_CHECK(mg_params(rb, rb_ref));

  MGaussian rd_front = f.restrict_dims(front_dims, front_vals);
  BOOST_CHECK(mg_properties(rd_front, Shape({1})));
  BOOST_CHECK(mg_params(rd_front, rf_ref));

  MGaussian rd_back = f.restrict_dims(back_dims, back_vals);
  BOOST_CHECK(mg_properties(rd_back, Shape({2})));
  BOOST_CHECK(mg_params(rd_back, rb_ref));
}

BOOST_AUTO_TEST_CASE(test_multiply_front_back_dims) {
  MGaussian cond(Shape{2, 1},
                 Vec{{1.0, -0.5}},
                 Mat{{1.5, 0.2}, {0.2, 2.0}},
                 Mat{{0.7}, {-1.2}},
                 0.3);
  MGaussian prior(Shape{1},
                  Vec{{0.4}},
                  Mat{{1.8}},
                  -0.2);

  MGaussian back = cond.multiply_back(prior);
  MGaussian front = prior.multiply_front(cond);
  MGaussian ref = cond.canonical().multiply_back(prior.canonical()).moment();

  BOOST_CHECK(mg_properties(back, Shape({2, 1})));
  BOOST_CHECK(mg_properties(front, Shape({2, 1})));
  BOOST_CHECK(mg_params(back, ref));
  BOOST_CHECK(mg_params(front, ref));

  MGaussian a(Shape{2, 1},
              Vec{{0.5, -1.0, 0.7}},
              Mat{{2.0, 0.2, -0.1}, {0.2, 1.7, 0.3}, {-0.1, 0.3, 1.5}},
              0.1);
  MGaussian b(Shape{1, 2},
              Vec{{-0.3, 0.8, 1.2}},
              Mat{{1.9, 0.1, -0.2}, {0.1, 2.2, 0.4}, {-0.2, 0.4, 1.6}},
              -0.4);

  Dims i = make_dims({0, 2});
  Dims j = make_dims({1, 3});
  MGaussian dims = a.multiply(b, i, j);
  MGaussian dims_ref = multiply(a.canonical(), b.canonical(), i, j).moment();

  BOOST_CHECK(mg_properties(dims, Shape({2, 1, 1, 2})));
  BOOST_CHECK(mg_params(dims, dims_ref));
}

BOOST_AUTO_TEST_CASE(test_entropy_and_kl) {
  Vec mean{{2.0, 0.5, 0.2}};
  Mat cov{{2.0, 1.0, 1.0}, {1.0, 3.0, 1.0}, {1.0, 1.0, 4.0}};
  MGaussian p(Shape{3}, mean, cov, 2.0);

  const double l2pi = std::log(boost::math::constants::two_pi<double>());
  const double ent = std::log(cov.determinant()) + 3.0 * (l2pi + 1.0);
  BOOST_CHECK_CLOSE(p.entropy(), ent / 2.0, 1e-5);

  MGaussian q(Shape{3}, mean, cov + Mat::Identity(3, 3));
  BOOST_CHECK_GE(p.kl_divergence(q), 0.0);
  BOOST_CHECK_SMALL(p.kl_divergence(p), 1e-6);
}

BOOST_AUTO_TEST_CASE(test_maximum_family) {
  MGaussian f(Shape{1, 1, 1},
              Vec{{0.2, -0.7, 1.1}},
              Mat{{2.4, 0.5, -0.2}, {0.5, 1.8, 0.3}, {-0.2, 0.3, 2.1}},
              0.6);

  Vec argmax_mg;
  Vec argmax_cg;
  Exp<double> mx = f.maximum(&argmax_mg);
  Exp<double> mx_ref = f.canonical().maximum(&argmax_cg);
  BOOST_CHECK_CLOSE(mx.lv, mx_ref.lv, 1e-8);
  BOOST_CHECK(argmax_mg.isApprox(argmax_cg, 1e-8));

  MGaussian mf = f.maximum_front(2);
  MGaussian mb = f.maximum_back(2);
  MGaussian md = f.maximum_dims(make_dims({0, 2}));

  MGaussian mf_ref = f.canonical().maximum_front(2).moment();
  MGaussian mb_ref = f.canonical().maximum_back(2).moment();
  MGaussian md_ref = f.canonical().maximum_dims(make_dims({0, 2})).moment();

  BOOST_CHECK(mg_properties(mf, Shape({1, 1})));
  BOOST_CHECK(mg_properties(mb, Shape({1, 1})));
  BOOST_CHECK(mg_properties(md, Shape({1, 1})));

  BOOST_CHECK(mg_params(mf, mf_ref));
  BOOST_CHECK(mg_params(mb, mb_ref));
  BOOST_CHECK(mg_params(md, md_ref));
}

BOOST_AUTO_TEST_CASE(test_consistency_with_canonical_gaussian) {
  const Vec eta{{1.2, -0.4, 0.8}};
  const Mat lambda{{4.0, 0.3, -0.2}, {0.3, 3.5, 0.4}, {-0.2, 0.4, 2.8}};
  CGaussian cg(Shape{3}, eta, lambda, 1.5);

  MGaussian mg = cg.moment();
  CGaussian cg2 = mg.canonical();

  Vec x{{0.3, -1.0, 0.7}};
  BOOST_CHECK_CLOSE(mg.log(x), cg.log(x), 1e-8);
  BOOST_CHECK_CLOSE(mg(x).lv, cg(x).lv, 1e-8);

  BOOST_CHECK_CLOSE(mg.marginal().lv, cg.marginal().lv, 1e-8);
  BOOST_CHECK(cg_params(cg2, cg.inf_vector(), cg.inf_matrix(), cg.log_multiplier()));
}
