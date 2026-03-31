#define BOOST_TEST_MODULE grid_argument
#include <boost/test/unit_test.hpp>

#include <libgm/argument/grid_argument.hpp>

#include <unordered_set>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_grid_arg_defaults_and_ordering) {
  using Arg = GridArg;

  Arg a;
  Arg b{0, 1};
  Arg c{1, 0};

  BOOST_CHECK(a == (Arg{0, 0}));
  BOOST_CHECK(a < b);
  BOOST_CHECK(b < c);
}

BOOST_AUTO_TEST_CASE(test_grid_arg_hash_and_unordered_set) {
  using Arg = GridArg;

  Arg x{1, 2};
  Arg y{1, 2};
  Arg z{2, 1};

  BOOST_CHECK(x == y);
  BOOST_CHECK(std::hash<Arg>()(x) == std::hash<Arg>()(y));

  std::unordered_set<Arg> set;
  set.insert(x);
  BOOST_CHECK(set.contains(y));
  BOOST_CHECK(!set.contains(z));
}
