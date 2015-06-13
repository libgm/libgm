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
  BOOST_CHECK_EQUAL(num_values(x), 3);
  BOOST_CHECK(is_discrete(x));
  BOOST_CHECK(!is_continuous(x));

  BOOST_CHECK_EQUAL(y.name(), "y");
  BOOST_CHECK_EQUAL(num_dimensions(y), 2);
  BOOST_CHECK(is_continuous(y));
  BOOST_CHECK(!is_discrete(y));

  BOOST_CHECK_EQUAL(z.name(), "z");
  BOOST_CHECK_EQUAL(num_values(z), 2);
  BOOST_CHECK(is_discrete(z));
  BOOST_CHECK(!is_continuous(z));

  BOOST_CHECK_EQUAL(q.name(), "q");
  BOOST_CHECK_EQUAL(num_dimensions(q), 2);
  BOOST_CHECK(is_continuous(q));
  BOOST_CHECK(!is_discrete(q));
}
