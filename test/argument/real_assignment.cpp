#define BOOST_TEST_MODULE real_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/argument/real_assignment.hpp>
#include <libgm/argument/universe.hpp>

#include "../math/eigen/helpers.hpp"

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_num_dimensions) {
  universe u;
  variable x = u.new_continuous_variable("x", 3);
  variable y = u.new_continuous_variable("y", 2);

  real_assignment<double> a;
  BOOST_CHECK_EQUAL(num_dimensions(a), 0);

  a[x] = vec3(1, 2, 3);
  a[y] = vec2(2, 1);
  BOOST_CHECK_EQUAL(num_dimensions(a), 5);
}

BOOST_AUTO_TEST_CASE(test_extract) {
  universe u;
  variable x = u.new_continuous_variable("x", 3);
  variable y = u.new_continuous_variable("y", 2);

  real_assignment<double> a;
  a[x] = vec3(1, 2, 3);
  a[y] = vec2(2, 1);
  BOOST_CHECK_EQUAL(extract(a, {y, x}), vec5(2, 1, 1, 2, 3));
  BOOST_CHECK_EQUAL(extract(a, {x}), vec3(1, 2, 3));
}
