#define BOOST_TEST_MODULE logarithmic_vector
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/logarithmic_vector.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/experimental/logarithmic_table.hpp>
#include <libgm/factor/experimental/probability_vector.hpp>

#include "../predicates.hpp"

namespace libgm { namespace experimental {
  template class logarithmic_vector<var, double>;
} }

using namespace libgm;

typedef experimental::logarithmic_vector<var> lvector;
typedef experimental::probability_vector<var> pvector;
typedef experimental::logarithmic_table<var> ltable;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lvector a(y);
  BOOST_CHECK(table_properties(a, {y}));

  lvector b({x}, logd(3.0));
  BOOST_CHECK(table_properties(b, {x}));
  BOOST_CHECK_CLOSE(b[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(b[1], std::log(3.0), 1e-8);

  lvector c({y}, Eigen::Vector3d(2.0, 3.0, 4.0));
  BOOST_CHECK(table_properties(c, {y}));
  BOOST_CHECK_CLOSE(c[0], 2.0, 1e-8);
  BOOST_CHECK_CLOSE(c[1], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(c[2], 4.0, 1e-8);

  lvector d({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_EQUAL(d[0], 6.0);
  BOOST_CHECK_EQUAL(d[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lvector f = pvector({x}, {0.5, 0.7}).logarithmic();
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.7), 1e-8);

  lvector g = ltable({y}, {0.1, 0.2, 0.3}).vector();
  BOOST_CHECK(table_properties(g, {y}));
  BOOST_CHECK_EQUAL(g[0], 0.1);
  BOOST_CHECK_EQUAL(g[1], 0.2);
  BOOST_CHECK_EQUAL(g[2], 0.3);

  swap(f, g);
  BOOST_CHECK(table_properties(f, y));
  BOOST_CHECK(table_properties(g, x));
}

BOOST_AUTO_TEST_CASE(test_transform) {
  universe u;
  var x = var::discrete(u, "x", 3);

  lvector f(x, {1, 2, 3});
  lvector g(x, {0.5, 1.0, 1.5});
  lvector h(x, {1.0, 1.0, 4.5});

  logd one(1.0, log_tag());
  logd half(0.5, log_tag());
  logd four(4.0, log_tag());

  // Unary transforms
  BOOST_CHECK_SMALL(max_diff(f * one, lvector(x, {2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(one * f, lvector(x, {2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / half, lvector(x, {0.5, 1.5, 2.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(four / f, lvector(x, {3.0, 2.0, 1.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), lvector(x, {2.0, 4.0, 6.0})), 1e-8);
  // Binary transforms
  BOOST_CHECK_SMALL(max_diff(f * g, lvector(x, {1.5, 3.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, lvector(x, {0.5, 1.0, 1.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(max(f, h), lvector(x, {1.0, 2.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(min(f, h), lvector(x, {1.0, 1.0, 3.0})), 1e-8);

  // Left compositions
  BOOST_CHECK_SMALL(max_diff((f*g) / h, lvector(x, {0.5, 2.0, 0.0})), 1e-8);

  // Right compositions
  BOOST_CHECK_SMALL(max_diff(f / (g*h), lvector(x, {-0.5, 0.0, -3.0})), 1e-8);

  // Multi-way compositions
  BOOST_CHECK_SMALL(max_diff(max(f,h)/min(g,h), lvector(x, {0.5, 1.0, 3.0})), 1e-8);

  // Transform of a transform
  BOOST_CHECK_SMALL(max_diff((f * one) / half, lvector(x, {1.5, 2.5, 3.5})), 1e-8);
}
