#define BOOST_TEST_MODULE uint_assignment_iterator
#include <boost/test/unit_test.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class uint_assignment_iterator<var>;
  template class uint_assignment_iterator<vec>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_assignment_iterator<var>);
BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_assignment_iterator<vec>);

BOOST_AUTO_TEST_CASE(test_univariate) {
  universe u;
  var x = var::discrete(u, "x", 3);
  var y = var::discrete(u, "y", 2);

  // test a non-empty domain
  uint_assignment_iterator<var> it({y, x}), end;
  uint_assignment<var> a;
  for (std::size_t i = 0; i < 3; ++i) {
    a[x] = i;
    for (std::size_t j = 0; j < 2; ++j) {
      a[y] = j;
      BOOST_CHECK_NE(it, end);
      BOOST_CHECK_EQUAL(*it++, a);
    }
  }
  BOOST_CHECK_EQUAL(it, end);

  // test an empty domain
  it = uint_assignment_iterator<var>(domain<var>());
  a.clear();
  BOOST_CHECK_EQUAL(*it, a);
  BOOST_CHECK_EQUAL(++it, end);
}

BOOST_AUTO_TEST_CASE(test_multivariate) {
  universe u;
  vec x = vec::discrete(u, "x", {2, 3});
  vec y = vec::discrete(u, "y", 2);

  // test a non-empty domain
  uint_assignment_iterator<vec> it({y, x}), end;
  uint_assignment<vec> a;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      a[x] = {j, i};
      for (std::size_t k = 0; k < 2; ++k) {
        a[y] = {k};
        BOOST_CHECK_NE(it, end);
        BOOST_CHECK_EQUAL(*it++, a);
      }
    }
  }
  BOOST_CHECK_EQUAL(it, end);

  // test an empty domain
  it = uint_assignment_iterator<vec>(domain<vec>());
  a.clear();
  BOOST_CHECK_EQUAL(*it, a);
  BOOST_CHECK_EQUAL(++it, end);
}
