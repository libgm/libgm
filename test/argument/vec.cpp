#define BOOST_TEST_MODULE vec
#include <boost/test/unit_test.hpp>

#include <libgm/argument/vec.hpp>
#include <libgm/datastructure/uint_vector.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_comparisons) {
  universe u;
  vec x = vec::continuous(u, "x", 2);
  vec y = vec::continuous(u, "y", 3);

  BOOST_CHECK_EQUAL(x.desc()->name, "x");

  BOOST_CHECK(x == x);
  BOOST_CHECK(x != y);
  BOOST_CHECK((x < y) ^ (y < x));
  BOOST_CHECK((x > y) ^ (y > x));
  BOOST_CHECK((x >= y) || (x <= y));
}

BOOST_AUTO_TEST_CASE(test_traits) {
  universe u;
  vec r = vec::continuous(u, "r", 1);
  vec x = vec::continuous(u, "x", 2);
  vec y = vec::continuous(u, "y", 2);
  vec z = vec::discrete(u, "z", {3, 4});
  vec w = vec::discrete(u, "w", {3, 4});
  vec q = vec::discrete(u, "q", {3, 2});

  BOOST_CHECK(!vec::compatible(r, x));
  BOOST_CHECK(vec::compatible(x, y));
  BOOST_CHECK(!vec::compatible(x, z));
  BOOST_CHECK(vec::compatible(z, w));
  BOOST_CHECK(!vec::compatible(z, q));

  BOOST_CHECK_THROW(x.num_values(), std::invalid_argument);
  BOOST_CHECK_THROW(x.num_values(0), std::invalid_argument);
  BOOST_CHECK_EQUAL(z.num_values(), 12);
  BOOST_CHECK_EQUAL(z.num_values(0), 3);
  BOOST_CHECK(x.continuous());
  BOOST_CHECK(z.discrete());
  BOOST_CHECK(!x.indexed());
  BOOST_CHECK(!z.indexed());
}
