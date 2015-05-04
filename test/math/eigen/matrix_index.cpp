#define BOOST_TEST_MODULE matrix_index
#include <boost/test/unit_test.hpp>

#include <libgm/math/eigen/matrix_index.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_constructors) {
  matrix_index a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK_EQUAL(a.start(), 0);

  matrix_index b(2, 4);
  BOOST_CHECK(!b.empty());
  BOOST_CHECK(b.contiguous());
  BOOST_CHECK_EQUAL(b.start(), 2);
  BOOST_CHECK_EQUAL(b.size(), 4);

  matrix_index c = {3, 4};
  BOOST_CHECK(!c.empty());
  BOOST_CHECK(!c.contiguous());
  BOOST_CHECK_EQUAL(c.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_append) {
  matrix_index a;
  a.append(2, 0);
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.contiguous());

  a.append(1, 3);
  BOOST_CHECK(!a.empty());
  BOOST_CHECK(a.contiguous());
  BOOST_CHECK_EQUAL(a.start(), 1);
  BOOST_CHECK_EQUAL(a.size(), 3);
  BOOST_CHECK_EQUAL(a(1), 2);

  a.append(4, 2);
  BOOST_CHECK(!a.empty());
  BOOST_CHECK(a.contiguous());
  BOOST_CHECK_EQUAL(a.start(), 1);
  BOOST_CHECK_EQUAL(a.size(), 5);
  BOOST_CHECK_EQUAL(a(1), 2);

  a.append(0, 1);
  BOOST_CHECK(!a.empty());
  BOOST_CHECK(!a.contiguous());
  BOOST_CHECK_EQUAL(a.start(), 0);
  BOOST_CHECK_EQUAL(a.size(), 6);
  BOOST_CHECK_EQUAL(a(1), 2);
  BOOST_CHECK_EQUAL(a[1], 2);
  BOOST_CHECK_EQUAL(a(5), 0);
  BOOST_CHECK_EQUAL(a[5], 0);

  a.append(10, 2);
  BOOST_CHECK(!a.empty());
  BOOST_CHECK(!a.contiguous());
  BOOST_CHECK_EQUAL(a.start(), 0);
  BOOST_CHECK_EQUAL(a.size(), 8);
  BOOST_CHECK_EQUAL(a(7), 11);
  BOOST_CHECK_EQUAL(a[7], 11);
}
