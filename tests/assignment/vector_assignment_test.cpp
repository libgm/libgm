#define BOOST_TEST_MODULE vector_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/assignment/vector_assignment.hpp>
#include <libgm/argument/named_argument.hpp>

#include <algorithm>

using namespace libgm;

namespace {

Arg make_arg(const char* name) {
  return NamedFactory::default_factory().make(name);
}

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
  Arg x = make_arg("x");
  Arg y = make_arg("y");
  Arg z = make_arg("z");
  Arg w = make_arg("w");
  Arg q = make_arg("q");

  VectorAssignment<double> a = {
    {x, vec({1, 2, 3})},
    {y, vec({0, 1})},
    {z, vec({5})},
  };

  BOOST_CHECK(equal_exact(a.at(x), vec({1, 2, 3})));
  BOOST_CHECK(equal_exact(a.at(y), vec({0, 1})));
  BOOST_CHECK(equal_exact(a.at(z), vec({5})));

  VectorAssignment<double> b(a.begin(), a.end());
  VectorAssignment<double> c;
  c.set(x, vec({1, 2, 3}));
  c.set(y, vec({0, 1}));
  c.set(z, vec({5}));
  BOOST_CHECK(a == b);
  BOOST_CHECK(a == c);

  BOOST_CHECK(equal_exact(a.values(z), vec({5})));
  BOOST_CHECK(equal_exact(a.values(Domain({z, x})), vec({5, 1, 2, 3})));

  c.set(w, vec({4}));
  c.set(y, vec({7, 8}));
  c.set(q, vec({9}));
  BOOST_CHECK(equal_exact(c.at(w), vec({4})));
  BOOST_CHECK(equal_exact(c.at(y), vec({7, 8})));
  BOOST_CHECK(equal_exact(c.at(q), vec({9})));

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

  VectorAssignment<double> a = {
    {z, vec({5})},
    {x, vec({1, 2, 3})},
    {y, vec({0, 1})},
  };
  BOOST_CHECK(a.keys() == Domain({x, y, z}));

  BOOST_CHECK_THROW(a.at(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(q), std::out_of_range);
  BOOST_CHECK_THROW(a.values(Domain({x, q})), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_empty_domain_and_zero_length_values) {
  Arg x = make_arg("x");
  Arg r = make_arg("r");
  Arg w = make_arg("w");

  VectorAssignment<double> a = {
    {x, vec({1, 2, 3})},
    {r, vec({})},
  };

  Vector<double> empty = a.values(Domain());
  BOOST_CHECK_EQUAL(empty.size(), 0);
  BOOST_CHECK_EQUAL(a.values(r).size(), 0);

  BOOST_CHECK(equal_exact(a.values(Domain({r, x})), vec({1, 2, 3})));

  Domain present, absent;
  a.partition(Domain({x, w, x, r}), present, absent);
  BOOST_CHECK(present == Domain({x, x, r}));
  BOOST_CHECK(absent == Domain({w}));

  present.clear();
  absent.clear();
  a.partition(Domain(), present, absent);
  BOOST_CHECK(present.empty());
  BOOST_CHECK(absent.empty());
}
