#define BOOST_TEST_MODULE range_adaptors
#include <boost/test/unit_test.hpp>

#include <libgm/range/transformed.hpp>
#include <libgm/range/reversed.hpp>
#include <libgm/range/joined.hpp>

#include <boost/range/algorithm.hpp>

#include <vector>

int plus_one(int x) {
  return x + 1;
}

BOOST_AUTO_TEST_CASE(test_all) {
  using namespace libgm;

  std::vector<int> a = {0, 2, 4, 8};
  std::vector<int> b = {0, 2};

  std::vector<int> a1 = {1, 3, 5, 9};
  std::vector<int> ar = {8, 4, 2, 0};
  std::vector<int> ab = {0, 2, 4, 8, 0, 2};

  BOOST_CHECK(boost::equal(a1, make_transformed(a, plus_one)));
  BOOST_CHECK(boost::equal(ar, make_reversed(a)));
  BOOST_CHECK(boost::equal(ab, make_joined(a, b)));
}
