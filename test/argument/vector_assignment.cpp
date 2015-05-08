#define BOOST_TEST_MODULE vector_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/argument/vector_assignment.hpp>
#include <libgm/argument/universe.hpp>

#include "../math/eigen/helpers.hpp"

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_vector_size) {
  universe u;
  variable x = u.new_vector_variable("x", 3);
  variable y = u.new_vector_variable("y", 2);

  vector_assignment<double> a;
  BOOST_CHECK_EQUAL(vector_size(a), 0);

  a[x] = vec3(1, 2, 3);
  a[y] = vec2(2, 1);
  BOOST_CHECK_EQUAL(vector_size(a), 5);
}

BOOST_AUTO_TEST_CASE(test_extract) {
  universe u;
  variable x = u.new_vector_variable("x", 3);
  variable y = u.new_vector_variable("y", 2);

  vector_assignment<double> a;
  a[x] = vec3(1, 2, 3);
  a[y] = vec2(2, 1);
  BOOST_CHECK_EQUAL(extract(a, {y, x}), vec5(2, 1, 1, 2, 3));
  BOOST_CHECK_EQUAL(extract(a, {x}), vec3(1, 2, 3));
}
