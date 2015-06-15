#define BOOST_TEST_MODULE variable
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_constructors) {
  // Create a universe.
  universe u;

  // Create some variables.
  variable x = u.new_discrete_variable("x", 3);
  variable y = u.new_continuous_variable("y", 2);
  variable z = u.new_discrete_variable("z", 2);
  variable q = u.new_continuous_variable("q", 2);

  BOOST_CHECK_EQUAL(x.name(), "x");
  BOOST_CHECK_EQUAL(x.num_values(), 3);
  BOOST_CHECK(x.is_discrete());
  BOOST_CHECK(!x.is_continuous());

  BOOST_CHECK_EQUAL(y.name(), "y");
  BOOST_CHECK_EQUAL(y.num_dimensions(), 2);
  BOOST_CHECK(y.is_continuous());
  BOOST_CHECK(!y.is_discrete());

  BOOST_CHECK_EQUAL(z.name(), "z");
  BOOST_CHECK_EQUAL(z.num_values(), 2);
  BOOST_CHECK(z.is_discrete());
  BOOST_CHECK(!z.is_continuous());

  BOOST_CHECK_EQUAL(q.name(), "q");
  BOOST_CHECK_EQUAL(q.num_dimensions(), 2);
  BOOST_CHECK(q.is_continuous());
  BOOST_CHECK(!q.is_discrete());
}
