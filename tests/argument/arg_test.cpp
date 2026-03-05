#define BOOST_TEST_MODULE arg
#include <boost/test/unit_test.hpp>

#include <libgm/argument/argument.hpp>
#include <libgm/argument/named_argument.hpp>

#include <sstream>
#include <unordered_set>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_null_and_stream) {
  Arg a;
  BOOST_CHECK(!a);
  BOOST_CHECK(a == Arg(nullptr));

  std::ostringstream out;
  out << a;
  BOOST_CHECK_EQUAL(out.str(), "null");
}

BOOST_AUTO_TEST_CASE(test_comparisons_and_stream) {
  NamedFactory& f = NamedFactory::default_factory();
  Arg x = f.make("x");
  Arg y = f.make("y");

  BOOST_CHECK(x);
  BOOST_CHECK(y);
  BOOST_CHECK(x == x);
  BOOST_CHECK(y == y);
  BOOST_CHECK(x != y);
  BOOST_CHECK(x < y);
  BOOST_CHECK(y > x);

  std::ostringstream out;
  out << x;
  BOOST_CHECK_EQUAL(out.str(), "x");
}

BOOST_AUTO_TEST_CASE(test_hash_and_unordered_set) {
  NamedFactory& f = NamedFactory::default_factory();
  Arg x1 = f.make("x");
  Arg x2 = f.make("x");
  Arg y = f.make("y");

  BOOST_CHECK(x1 == x2);
  BOOST_CHECK(std::hash<Arg>()(x1) == std::hash<Arg>()(x2));

  std::unordered_set<Arg> set;
  set.insert(x1);
  BOOST_CHECK(set.contains(x2));
  BOOST_CHECK(!set.contains(y));
}

BOOST_AUTO_TEST_CASE(test_null_ordering_against_non_null) {
  NamedFactory& f = NamedFactory::default_factory();
  Arg n;
  Arg x = f.make("x");

  BOOST_CHECK(n < x);
}
