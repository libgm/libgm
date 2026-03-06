#define BOOST_TEST_MODULE subvector
#include <boost/test/unit_test.hpp>

#include <libgm/argument/span.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/subvector.hpp>

using namespace libgm;

using Vec = Vector<double>;

BOOST_AUTO_TEST_CASE(test_operations) {
  Vec v{{4, 3, 2, 1}};
  Spans s = {{0, 1}, {2, 2}}; // indices [0] and [2,3]

  auto sv = sub(v, s);
  BOOST_CHECK_EQUAL(sv.rows(), 3);
  BOOST_CHECK_EQUAL(sv.cols(), 1);
  BOOST_CHECK_EQUAL(sv.size(), 3); // should match total selected length
  BOOST_CHECK((Vec(sv)).isApprox(Vec{{4, 2, 1}}, 1e-8));

  Vec w{{-1, 1, 2}};
  BOOST_CHECK_CLOSE(sv.dot(w), 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_update) {
  const Vec v{{4, 3, 2, 1}};
  Spans a = {{0, 1}, {2, 1}}; // [0,2]
  Spans b = {{2, 2}};         // [2,3]

  Vec w = sub(v, a);
  BOOST_CHECK(w.isApprox(Vec{{4, 2}}, 1e-8));
  w += sub(v, a);
  BOOST_CHECK(w.isApprox(Vec{{8, 4}}, 1e-8));
  w -= sub(v, b);
  BOOST_CHECK(w.isApprox(Vec{{6, 3}}, 1e-8));

  Vec u = Vec::Zero(4);
  sub(u, a) = w;
  BOOST_CHECK(u.isApprox(Vec{{6, 0, 3, 0}}, 1e-8));
  sub(u, a) += w;
  BOOST_CHECK(u.isApprox(Vec{{12, 0, 6, 0}}, 1e-8));
  sub(u, b) -= w;
  BOOST_CHECK(u.isApprox(Vec{{12, 0, 0, -3}}, 1e-8));
}
