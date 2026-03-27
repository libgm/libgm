#define BOOST_TEST_MODULE mixture
#include <boost/test/unit_test.hpp>

#include <libgm/factor/mixture.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/factor/impl/moment_gaussian.hpp>

#include "predicates.hpp"

using namespace libgm;

using MGaussian = MomentGaussian<double>;
using GaussianMixture = Mixture<MGaussian>;
using Vec = Vector<double>;
using Mat = Matrix<double>;

BOOST_AUTO_TEST_CASE(test_constructors_and_accessors) {
  GaussianMixture a;
  BOOST_CHECK_EQUAL(a.size(), 0);
  BOOST_CHECK_EQUAL(a.arity(), 0);

  GaussianMixture b(3, Shape{2, 1});
  BOOST_CHECK_EQUAL(b.size(), 3);
  BOOST_CHECK_EQUAL(b.arity(), 2);
  BOOST_CHECK_EQUAL(b.shape(), Shape({2, 1}));
  BOOST_CHECK(mg_properties(b[0], Shape({2, 1}), 0));

  MGaussian base(Shape{2}, Vec{{1.0, -0.5}}, Mat{{2.0, 0.3}, {0.3, 1.5}}, 0.2);
  GaussianMixture c(2, base);
  BOOST_CHECK_EQUAL(c.size(), 2);
  BOOST_CHECK(mg_params(c[0], base));
  BOOST_CHECK(mg_params(c[1], base));
}

BOOST_AUTO_TEST_CASE(test_query_and_marginal_dims) {
  GaussianMixture m(2, Shape{1, 2});
  m[0] = MGaussian(Shape{1, 2},
                   Vec{{1.0, -0.3, 0.4}},
                   Mat{{2.0, 0.2, -0.1}, {0.2, 1.5, 0.3}, {-0.1, 0.3, 1.8}},
                   0.3);
  m[1] = MGaussian(Shape{1, 2},
                   Vec{{-0.7, 0.5, 1.2}},
                   Mat{{1.4, -0.2, 0.1}, {-0.2, 2.1, 0.4}, {0.1, 0.4, 1.7}},
                   -0.4);

  Vec values{{0.25, -0.3, 1.1}};
  const double expected = static_cast<double>(m[0](values)) + static_cast<double>(m[1](values));
  BOOST_CHECK_CLOSE(static_cast<double>(m(values)), expected, 1e-8);
  BOOST_CHECK_CLOSE(m.log(values), std::log(expected), 1e-8);

  GaussianMixture md = m.marginal_dims(make_dims({1}));
  BOOST_CHECK_EQUAL(md.size(), 2);
  BOOST_CHECK(mg_properties(md[0], Shape({2})));
  BOOST_CHECK(mg_properties(md[1], Shape({2})));
  BOOST_CHECK(mg_params(md[0], m[0].marginal_dims(make_dims({1}))));
  BOOST_CHECK(mg_params(md[1], m[1].marginal_dims(make_dims({1}))));
}

BOOST_AUTO_TEST_CASE(test_restrict_and_normalize) {
  GaussianMixture m(2, Shape{2, 1});
  m[0] = MGaussian(Shape{2, 1},
                   Vec{{0.4, -1.2, 0.7}},
                   Mat{{1.7, 0.2, -0.1}, {0.2, 2.1, 0.3}, {-0.1, 0.3, 1.4}},
                   std::log(2.0));
  m[1] = MGaussian(Shape{2, 1},
                   Vec{{-0.6, 0.8, -0.2}},
                   Mat{{2.4, -0.1, 0.2}, {-0.1, 1.6, -0.4}, {0.2, -0.4, 1.9}},
                   std::log(3.0));

  Vec y{{0.7}};
  GaussianMixture r = m.restrict_back(y);
  BOOST_CHECK_EQUAL(r.size(), 2);
  BOOST_CHECK(mg_properties(r[0], Shape({2})));
  BOOST_CHECK(mg_properties(r[1], Shape({2})));
  BOOST_CHECK(mg_params(r[0], m[0].restrict_back(y)));
  BOOST_CHECK(mg_params(r[1], m[1].restrict_back(y)));

  GaussianMixture n = m;
  const double total = static_cast<double>(m.marginal());
  n.normalize();
  BOOST_CHECK_CLOSE(static_cast<double>(n.marginal()), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(n[0].log_multiplier(), m[0].log_multiplier() - std::log(total), 1e-8);
  BOOST_CHECK_CLOSE(n[1].log_multiplier(), m[1].log_multiplier() - std::log(total), 1e-8);
}
