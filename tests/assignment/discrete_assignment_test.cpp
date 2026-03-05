#define BOOST_TEST_MODULE discrete_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/assignment/discrete_assignment.hpp>
#include <libgm/argument/named_argument.hpp>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(test_basic_operations) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");
  Arg w = make_arg("w");
  Arg q = make_arg("q");

  DiscreteAssignment a = {{x, 3}, {y, 2}, {z, 1}};
  BOOST_CHECK_EQUAL(a.at(x), 3);
  BOOST_CHECK_EQUAL(a.at(y), 2);
  BOOST_CHECK_EQUAL(a.at(z), 1);

  DiscreteAssignment b(a.begin(), a.end());
  DiscreteAssignment c;
  c.set(Domain({x, y, z}), {3, 2, 1});
  BOOST_CHECK(a == b);
  BOOST_CHECK(a == c);

  std::vector<size_t> z_x = a.values(Domain({z, x}));
  BOOST_CHECK(z_x == std::vector<size_t>({1, 3}));
  BOOST_CHECK(a.values(y) == std::vector<size_t>({2}));

  c.set(Domain({w, z}), {2, 3});
  BOOST_CHECK_EQUAL(c.at(w), 2);
  BOOST_CHECK_EQUAL(c.at(z), 3);

  c.set(Domain({x, q}), {2, 1});
  BOOST_CHECK_EQUAL(c.at(x), 2);
  BOOST_CHECK_EQUAL(c.at(q), 1);

  c.erase(w);
  c.erase(q);
  Domain present, absent;
  c.partition(Domain({x, w, y, q}), present, absent);
  BOOST_CHECK(present == Domain({x, y}));
  BOOST_CHECK(absent == Domain({w, q}));
}

BOOST_AUTO_TEST_CASE(test_keys_and_missing_value_paths) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");
  Arg q = make_arg("q");

  DiscreteAssignment a = {{z, 1}, {x, 3}, {y, 2}};
  BOOST_CHECK(a.keys() == Domain({x, y, z}));

  BOOST_CHECK_THROW(a.at(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(Domain({x, q})), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_partition_duplicates_and_empty_domain) {
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg w = make_arg("w");

  DiscreteAssignment a = {{x, 3}, {y, 2}};

  Domain present, absent;
  a.partition(Domain({x, w, x, y}), present, absent);
  BOOST_CHECK(present == Domain({x, x, y}));
  BOOST_CHECK(absent == Domain({w}));

  present.clear();
  absent.clear();
  a.partition(Domain(), present, absent);
  BOOST_CHECK(present.empty());
  BOOST_CHECK(absent.empty());
}
