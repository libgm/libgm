#define BOOST_TEST_MODULE var
#include <boost/test/unit_test.hpp>

#include <libgm/argument/var.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_comparisons) {
  universe u;
  var x = var::continuous(u, "x");
  var y = var::continuous(u, "y");

  BOOST_CHECK_EQUAL(x.desc()->name, "x");

  BOOST_CHECK(x == x);
  BOOST_CHECK(x != y);
  BOOST_CHECK((x < y) ^ (y < x));
  BOOST_CHECK((x > y) ^ (y > x));
  BOOST_CHECK((x >= y) || (x <= y));
}

BOOST_AUTO_TEST_CASE(test_traits) {
  universe u;
  var x = var::continuous(u, "x");
  var y = var::continuous(u, "y");
  var z = var::discrete(u, "z", 3);
  var w = var::discrete(u, "w", 3);
  var q = var::discrete(u, "q", 2);

  BOOST_CHECK(var::compatible(x, y));
  BOOST_CHECK(!var::compatible(x, z));
  BOOST_CHECK(var::compatible(z, w));
  BOOST_CHECK(!var::compatible(z, q));

  BOOST_CHECK_THROW(x.num_values(), std::invalid_argument);
  BOOST_CHECK_EQUAL(z.num_values(), 3);
  BOOST_CHECK(x.continuous());
  BOOST_CHECK(z.discrete());
  BOOST_CHECK(!x.indexed());
  BOOST_CHECK(!z.indexed());
}
