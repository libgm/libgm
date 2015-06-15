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
  BOOST_CHECK_EQUAL(p.num_values(), 4);
  BOOST_CHECK_EQUAL(q.num_values(), 2);
}

BOOST_FIXTURE_TEST_CASE(test_variables, fixture) {
  universe u;

  BOOST_CHECK_EQUAL(p(5).num_values(), 4);
  BOOST_CHECK_EQUAL(p(5).index(), 5);
  BOOST_CHECK_EQUAL(p(5).name(), "p");
  BOOST_CHECK_EQUAL(q(8).num_values(), 2);
  BOOST_CHECK_EQUAL(q(8).index(), 8);
  BOOST_CHECK_EQUAL(q(8).name(), "q");
}
