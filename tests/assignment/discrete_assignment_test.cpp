#define BOOST_TEST_MODULE discrete_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/argument/named_argument.hpp>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;
using Assignment = libgm::DiscreteAssignment<Arg>;

} // namespace

BOOST_AUTO_TEST_CASE(test_basic_operations) {
  Arg x("x");
  Arg y("y");
  Arg z("z");
  Arg w("w");
  Arg q("q");

  Assignment a = {{x, 3}, {y, 2}, {z, 1}};
  BOOST_CHECK_EQUAL(a.at(x), 3);
  BOOST_CHECK_EQUAL(a.at(y), 2);
  BOOST_CHECK_EQUAL(a.at(z), 1);

  Assignment b(a.begin(), a.end());
  Assignment c;
  c.set(libgm::Domain<Arg>({x, y, z}), {3, 2, 1});
  BOOST_CHECK(a == b);
  BOOST_CHECK(a == c);

  std::vector<size_t> z_x = a.values(libgm::Domain<Arg>({z, x}));
  BOOST_CHECK(z_x == std::vector<size_t>({1, 3}));
  BOOST_CHECK(a.values(y) == std::vector<size_t>({2}));

  c.set(libgm::Domain<Arg>({w, z}), {2, 3});
  BOOST_CHECK_EQUAL(c.at(w), 2);
  BOOST_CHECK_EQUAL(c.at(z), 3);

  c.set(libgm::Domain<Arg>({x, q}), {2, 1});
  BOOST_CHECK_EQUAL(c.at(x), 2);
  BOOST_CHECK_EQUAL(c.at(q), 1);

  c.erase(w);
  c.erase(q);
  libgm::Domain<Arg> present, absent;
  c.partition(libgm::Domain<Arg>({x, w, y, q}), present, absent);
  BOOST_CHECK(present == libgm::Domain<Arg>({x, y}));
  BOOST_CHECK(absent == libgm::Domain<Arg>({w, q}));
}

BOOST_AUTO_TEST_CASE(test_keys_and_missing_value_paths) {
  Arg x("x");
  Arg y("y");
  Arg z("z");
  Arg q("q");

  Assignment a = {{z, 1}, {x, 3}, {y, 2}};
  BOOST_CHECK(a.keys() == libgm::Domain<Arg>({x, y, z}));

  BOOST_CHECK_THROW(a.at(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(libgm::Domain<Arg>({x, q})), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_partition_duplicates_and_empty_domain) {
  Arg x("x");
  Arg y("y");
  Arg w("w");

  Assignment a = {{x, 3}, {y, 2}};

  libgm::Domain<Arg> present, absent;
  a.partition(libgm::Domain<Arg>({x, w, x, y}), present, absent);
  BOOST_CHECK(present == libgm::Domain<Arg>({x, x, y}));
  BOOST_CHECK(absent == libgm::Domain<Arg>({w}));

  present.clear();
  absent.clear();
  a.partition(libgm::Domain<Arg>(), present, absent);
  BOOST_CHECK(present.empty());
  BOOST_CHECK(absent.empty());
}
