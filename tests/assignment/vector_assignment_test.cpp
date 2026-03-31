#define BOOST_TEST_MODULE vector_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/assignment/vector_assignment.hpp>
#include <libgm/argument/named_argument.hpp>

#include <algorithm>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;
using Assignment = libgm::VectorAssignment<Arg, double>;

Vector<double> vec(std::initializer_list<double> values) {
  Vector<double> result(values.size());
  std::copy(values.begin(), values.end(), result.data());
  return result;
}

bool equal_exact(const Vector<double>& a, const Vector<double>& b) {
  return a.size() == b.size() && (a.array() == b.array()).all();
}

} // namespace

BOOST_AUTO_TEST_CASE(test_basic_operations) {
  Arg x("x");
  Arg y("y");
  Arg z("z");
  Arg w("w");
  Arg q("q");

  Assignment a = {
    {x, vec({1, 2, 3})},
    {y, vec({0, 1})},
    {z, vec({5})},
  };

  BOOST_CHECK(equal_exact(a.at(x), vec({1, 2, 3})));
  BOOST_CHECK(equal_exact(a.at(y), vec({0, 1})));
  BOOST_CHECK(equal_exact(a.at(z), vec({5})));

  Assignment b(a.begin(), a.end());
  Assignment c;
  c.set(x, vec({1, 2, 3}));
  c.set(y, vec({0, 1}));
  c.set(z, vec({5}));
  BOOST_CHECK(a == b);
  BOOST_CHECK(a == c);

  BOOST_CHECK(equal_exact(a.values(z), vec({5})));
  BOOST_CHECK(equal_exact(a.values(libgm::Domain<Arg>({z, x})), vec({5, 1, 2, 3})));

  c.set(w, vec({4}));
  c.set(y, vec({7, 8}));
  c.set(q, vec({9}));
  BOOST_CHECK(equal_exact(c.at(w), vec({4})));
  BOOST_CHECK(equal_exact(c.at(y), vec({7, 8})));
  BOOST_CHECK(equal_exact(c.at(q), vec({9})));

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

  Assignment a = {
    {z, vec({5})},
    {x, vec({1, 2, 3})},
    {y, vec({0, 1})},
  };
  BOOST_CHECK(a.keys() == libgm::Domain<Arg>({x, y, z}));

  BOOST_CHECK_THROW(a.at(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(libgm::Domain<Arg>({x, q})), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_empty_domain_and_zero_length_values) {
  Arg x("x");
  Arg r("r");
  Arg w("w");

  Assignment a = {
    {x, vec({1, 2, 3})},
    {r, vec({})},
  };

  Vector<double> empty = a.values(libgm::Domain<Arg>());
  BOOST_CHECK_EQUAL(empty.size(), 0);
  BOOST_CHECK_EQUAL(a.values(r).size(), 0);

  BOOST_CHECK(equal_exact(a.values(libgm::Domain<Arg>({r, x})), vec({1, 2, 3})));

  libgm::Domain<Arg> present, absent;
  a.partition(libgm::Domain<Arg>({x, w, x, r}), present, absent);
  BOOST_CHECK(present == libgm::Domain<Arg>({x, x, r}));
  BOOST_CHECK(absent == libgm::Domain<Arg>({w}));

  present.clear();
  absent.clear();
  a.partition(libgm::Domain<Arg>(), present, absent);
  BOOST_CHECK(present.empty());
  BOOST_CHECK(absent.empty());
}
