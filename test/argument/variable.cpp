#define BOOST_TEST_MODULE variable
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_constructors) {
  // Create a universe.
  universe u;

  // Create some variables.
  variable x = u.new_finite_variable("x", 3);
  variable y = u.new_vector_variable("y", 2);
  variable z = u.new_finite_variable("z", 2);
  variable q = u.new_vector_variable("q", 2);

  BOOST_CHECK_EQUAL(x.name(), "x");
  BOOST_CHECK_EQUAL(x.size(), 3);
  BOOST_CHECK(x.finite());

  BOOST_CHECK_EQUAL(y.name(), "y");
  BOOST_CHECK_EQUAL(y.size(), 2);
  BOOST_CHECK(y.vector());

  BOOST_CHECK_EQUAL(z.name(), "z");
  BOOST_CHECK_EQUAL(z.size(), 2);
  BOOST_CHECK(z.finite());

  BOOST_CHECK_EQUAL(q.name(), "q");
  BOOST_CHECK_EQUAL(q.size(), 2);
  BOOST_CHECK(q.vector());
}
