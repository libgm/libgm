#define BOOST_TEST_MODULE logarithmic_vector
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_vector.hpp>

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"

namespace libgm {
  template class logarithmic_vector<double>;
  template class logarithmic_vector<float>;
}

using namespace libgm;

using lvector = logarithmic_vector<>;
using pvector = probability_vector<>;
using ltable = logarithmic_table<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  lvector a(3);
  BOOST_CHECK(vector_properties(a, 3));

  lvector b(2, logd(3.0));
  BOOST_CHECK(vector_properties(b, 2));
  BOOST_CHECK_CLOSE(b[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(b[1], std::log(3.0), 1e-8);

  lvector c(Eigen::Vector3d(2.0, 3.0, 4.0));
  BOOST_CHECK(vector_properties(c, 3));
  BOOST_CHECK_CLOSE(c[0], 2.0, 1e-8);
  BOOST_CHECK_CLOSE(c[1], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(c[2], 4.0, 1e-8);

  lvector d = {6.0, 6.5};
  BOOST_CHECK(vector_properties(d, 2));
  BOOST_CHECK_EQUAL(d[0], 6.0);
  BOOST_CHECK_EQUAL(d[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  lvector f = pvector({0.5, 0.7}).logarithmic();
  BOOST_CHECK(vector_properties(f, 2));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.7), 1e-8);

  lvector g = ltable({3}, {0.1, 0.2, 0.3}).vector();
  BOOST_CHECK(vector_properties(g, 3));
  BOOST_CHECK_EQUAL(g[0], 0.1);
  BOOST_CHECK_EQUAL(g[1], 0.2);
  BOOST_CHECK_EQUAL(g[2], 0.3);

  swap(f, g);
  BOOST_CHECK(vector_properties(f, 3));
  BOOST_CHECK(vector_properties(g, 2));
}

BOOST_AUTO_TEST_CASE(test_transform) {
  lvector f({1, 2, 3});
  lvector g({0.5, 1.0, 1.5});
  lvector h({1.0, 1.0, 4.5});

  logd one(1.0, log_tag());
  logd half(0.5, log_tag());
  logd four(4.0, log_tag());

  // Unary transforms
  BOOST_CHECK_SMALL(max_diff(f * one, lvector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(one * f, lvector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / half, lvector({0.5, 1.5, 2.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(four / f, lvector({3.0, 2.0, 1.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), lvector({2.0, 4.0, 6.0})), 1e-8);
  // Binary transforms
  BOOST_CHECK_SMALL(max_diff(f * g, lvector({1.5, 3.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, lvector({0.5, 1.0, 1.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(max(f, h), lvector({1.0, 2.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(min(f, h), lvector({1.0, 1.0, 3.0})), 1e-8);

  // Left compositions
  BOOST_CHECK_SMALL(max_diff((f*g) / h, lvector({0.5, 2.0, 0.0})), 1e-8);

  // Right compositions
  BOOST_CHECK_SMALL(max_diff(f / (g*h), lvector({-0.5, 0.0, -3.0})), 1e-8);

  // Multi-way compositions
  BOOST_CHECK_SMALL(max_diff(max(f,h)/min(g,h), lvector({0.5, 1.0, 3.0})), 1e-8);

  // Transform of a transform
  BOOST_CHECK_SMALL(max_diff((f * one) / half, lvector({1.5, 2.5, 3.5})), 1e-8);
}
