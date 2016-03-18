#define BOOST_TEST_MODULE probability_vector
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/probability_vector.hpp>

#include <libgm/factor/experimental/logarithmic_vector.hpp>
#include <libgm/factor/experimental/probability_table.hpp>

#include "predicates.hpp"

namespace libgm { namespace experimental {
  template class probability_vector<double>;
  template class probability_vector<float>;
} }

using namespace libgm;

typedef experimental::logarithmic_vector<> lvector;
typedef experimental::probability_vector<> pvector;
typedef experimental::probability_table<> ptable;

BOOST_AUTO_TEST_CASE(test_constructors) {
  pvector a(3);
  BOOST_CHECK(vector_properties(a, 3));

  pvector b(2, 3.0);
  BOOST_CHECK(vector_properties(b, 2));
  BOOST_CHECK_CLOSE(b[0], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(b[1], 3.0, 1e-8);

  pvector c(Eigen::Vector3d(2.0, 3.0, 4.0));
  BOOST_CHECK(vector_properties(c, 3));
  BOOST_CHECK_CLOSE(c[0], 2.0, 1e-8);
  BOOST_CHECK_CLOSE(c[1], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(c[2], 4.0, 1e-8);

  pvector d = {6.0, 6.5};
  BOOST_CHECK(vector_properties(d, 2));
  BOOST_CHECK_EQUAL(d[0], 6.0);
  BOOST_CHECK_EQUAL(d[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  pvector f = lvector({0.5, 0.7}).probability();
  BOOST_CHECK(vector_properties(f, 2));
  BOOST_CHECK_CLOSE(f[0], std::exp(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::exp(0.7), 1e-8);

  pvector g = ptable({3}, {0.1, 0.2, 0.3}).vector();
  BOOST_CHECK(vector_properties(g, 3));
  BOOST_CHECK_EQUAL(g[0], 0.1);
  BOOST_CHECK_EQUAL(g[1], 0.2);
  BOOST_CHECK_EQUAL(g[2], 0.3);

  swap(f, g);
  BOOST_CHECK(vector_properties(f, 3));
  BOOST_CHECK(vector_properties(g, 2));
}

BOOST_AUTO_TEST_CASE(test_transform) {
  pvector f({1, 2, 3});
  pvector g({0.5, 1.0, 1.5});
  pvector h({1.0, 1.0, 4.5});

  // Unary transforms
  BOOST_CHECK_SMALL(max_diff(f + 1.0, pvector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(1.0 + f, pvector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f - 0.5, pvector({0.5, 1.5, 2.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(4.0 - f, pvector({3.0, 2.0, 1.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f * 2.0, pvector({2.0, 4.0, 6.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(2.0 * f, pvector({2.0, 4.0, 6.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / 2.0, pvector({0.5, 1.0, 1.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(6.0 / f, pvector({6.0, 3.0, 2.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), pvector({1.0, 4.0, 9.0})), 1e-8);

  // Binary transforms
  BOOST_CHECK_SMALL(max_diff(f + g, pvector({1.5, 3.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f - g, pvector({0.5, 1.0, 1.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f * g, pvector({0.5, 2.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, pvector({2.0, 2.0, 2.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(max(f, h), pvector({1.0, 2.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(min(f, h), pvector({1.0, 1.0, 3.0})), 1e-8);

  // Left compositions
  BOOST_CHECK_SMALL(max_diff((f+g) / h, pvector({1.5, 3.0, 1.0})), 1e-8);

  // Right compositions
  BOOST_CHECK_SMALL(max_diff(f * (g+h), pvector({1.5, 4.0, 18.0})), 1e-8);

  // Multi-way compositions
  BOOST_CHECK_SMALL(max_diff((f*f)+(g*h), pvector({1.5, 5.0, 15.75})), 1e-8);

  // Transform of a transform
  BOOST_CHECK_SMALL(max_diff((f * 2.0) + 1.0, pvector({3.0, 5.0, 7.0})), 1e-8);
}
