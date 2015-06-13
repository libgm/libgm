#define BOOST_TEST_MODULE process
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>

using namespace libgm;

struct fixture {
  fixture()
    : p(u.new_discrete_dprocess("p", 4)),
      q(u.new_discrete_dprocess("q", 2)) { }
  universe u;
  dprocess p;
  dprocess q;
};

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  BOOST_CHECK_EQUAL(p.name(), "p");
  BOOST_CHECK_EQUAL(q.name(), "q");
  BOOST_CHECK_EQUAL(num_values(p), 4);
  BOOST_CHECK_EQUAL(num_values(q), 2);
}

BOOST_FIXTURE_TEST_CASE(test_variables, fixture) {
  universe u;

  BOOST_CHECK_EQUAL(num_values(p(5)), 4);
  BOOST_CHECK_EQUAL(index(p(5)), 5);
  BOOST_CHECK_EQUAL(p(5).name(), "p");
  BOOST_CHECK_EQUAL(num_values(q(8)), 2);
  BOOST_CHECK_EQUAL(index(q(8)), 8);
  BOOST_CHECK_EQUAL(q(8).name(), "q");
}
