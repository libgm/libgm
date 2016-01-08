#define BOOST_TEST_MODULE mixture
#include <boost/test/unit_test.hpp>

#include <libgm/model/mixture.hpp>

#include <libgm/argument/vec.hpp>
#include <libgm/factor/experimental/moment_gaussian.hpp>

#include "../math/eigen/helpers.hpp"

namespace libgm { namespace experimental {
  template class mixture<moment_gaussian<vec> >;
} }

using namespace libgm;

using mgaussian = experimental::moment_gaussian<vec>;
using gaussian_mixture = experimental::mixture<mgaussian>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  vec x = vec::continuous(u, "x", 2);
  vec y = vec::continuous(u, "y", 1);

  gaussian_mixture f({x, y}, 3);
  BOOST_CHECK_EQUAL(f.arguments(), domain<vec>({x, y}));
  BOOST_CHECK_EQUAL(f.arity(), 2);
  BOOST_CHECK_EQUAL(f.size(), 3);

  gaussian_mixture g(mgaussian({x}), 4);
  BOOST_CHECK_EQUAL(g.arguments(), domain<vec>({x}));
  BOOST_CHECK_EQUAL(g.arity(), 1);
  BOOST_CHECK_EQUAL(g.size(), 4);
}

BOOST_AUTO_TEST_CASE(test_queries) {
  universe u;
  vec x = vec::continuous(u, "x", 1);
  vec y = vec::continuous(u, "y", 2);

  gaussian_mixture f({x, y}, 2);
  f.param(0).mean << 1, 2, 3;
  f.param(1).mean << 4, 3, 2;
  f.param(0).cov << 2, 1, 1, 1, 2, 1, 1, 1, 4;
  f.param(1).cov << 3, 1, 1, 1, 2, 1, 1, 1, 3;

  gaussian_mixture g = f.marginal({y});
  BOOST_CHECK_EQUAL(g.factor(0), mgaussian({y}, vec2(2, 3), mat22(2, 1, 1, 4)));
  BOOST_CHECK_EQUAL(g.factor(1), mgaussian({y}, vec2(3, 2), mat22(2, 1, 1, 3)));
}

BOOST_AUTO_TEST_CASE(test_mutation) {
  universe u;
  vec x = vec::continuous(u, "x", 2);

  gaussian_mixture f({x}, 2);
  f.param(0).lm = std::log(1);
  f.param(1).lm = std::log(3);

  f.normalize();
  BOOST_CHECK_CLOSE(f.param(0).lm, std::log(0.25), 1e-6);
  BOOST_CHECK_CLOSE(f.param(1).lm, std::log(0.75), 1e-6);
}
