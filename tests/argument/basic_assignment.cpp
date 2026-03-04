#define BOOST_TEST_MODULE basic_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/argument/real_assignment.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/argument/universe.hpp>

#include "../math/eigen/helpers.hpp"

namespace libgm {
  template class basic_assignment<var, uint_vector>;
  template class basic_assignment<vec, uint_vector>;
  template class basic_assignment<var, dense_vector<>>;
  template class basic_assignment<vec, dense_vector<>>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);

BOOST_AUTO_TEST_CASE(test_univariate) {
  universe u;
  var x = var::discrete(u, "x", 6);
  var y = var::discrete(u, "y", 5);
  var z = var::discrete(u, "z", 4);
  var w = var::discrete(u, "w", 3);
  var q = var::discrete(u, "q", 2);

  uint_assignment<var> a({{x, 3}, {y, 2}, {z, 1}});
  BOOST_CHECK_EQUAL(a.at(x), 3);
  BOOST_CHECK_EQUAL(a.at(y), 2);
  BOOST_CHECK_EQUAL(a.at(z), 1);

  uint_assignment<var> b(a.begin(), a.end());
  uint_assignment<var> c({x, y, z}, {3, 2, 1});
  BOOST_CHECK_EQUAL(a, b);
  BOOST_CHECK_EQUAL(a, c);

  BOOST_CHECK_EQUAL(a.values({z, x}), uint_vector({1, 3}));

  BOOST_CHECK_EQUAL(a.insert({w, z}, {2, 3}), 1);
  BOOST_CHECK_EQUAL(a.at(w), 2);
  BOOST_CHECK_EQUAL(a.at(z), 1);

  BOOST_CHECK_EQUAL(a.insert_or_assign({x, q}, {2, 1}), 1);
  BOOST_CHECK_EQUAL(a.at(x), 2);
  BOOST_CHECK_EQUAL(a.at(q), 1);

  a.erase(w);
  a.erase(q);
  BOOST_CHECK(subset(domain<var>({x, y}), a));
  BOOST_CHECK(!subset(domain<var>({x, w}), a));
  BOOST_CHECK(disjoint(domain<var>({w, q}), a));
  BOOST_CHECK(!disjoint(domain<var>({w, x}), a));
}

BOOST_AUTO_TEST_CASE(test_multivariate) {
  universe u;
  vec x = vec::continuous(u, "x", 3);
  vec y = vec::continuous(u, "y", 2);
  vec z = vec::continuous(u, "z", 1);
  vec w = vec::continuous(u, "w", 1);
  vec q = vec::continuous(u, "q", 1);

  real_assignment<vec> a({{x, vec3(1, 2, 3)}, {y, vec2(0, 1)}, {z, vec1(5)}});
  BOOST_CHECK_EQUAL(a.at(x), vec3(1, 2, 3));
  BOOST_CHECK_EQUAL(a.at(y), vec2(0, 1));
  BOOST_CHECK_EQUAL(a.at(z), vec1(5));

  real_assignment<vec> b(a.begin(), a.end());
  real_assignment<vec> c({x, y, z}, vec6(1, 2, 3, 0, 1, 5));
  BOOST_CHECK_EQUAL(a, b);
  BOOST_CHECK_EQUAL(a, c);

  BOOST_CHECK_EQUAL(a.values({z, x}), vec4(5, 1, 2, 3));

  BOOST_CHECK_EQUAL(a.insert({w, z}, vec2(4, 3)), 1);
  BOOST_CHECK_EQUAL(a.at(w), vec1(4));
  BOOST_CHECK_EQUAL(a.at(z), vec1(5));

  BOOST_CHECK_EQUAL(a.insert_or_assign({y, q}, vec3(7, 8, 9)), 1);
  BOOST_CHECK_EQUAL(a.at(y), vec2(7, 8));
  BOOST_CHECK_EQUAL(a.at(q), vec1(9));

  a.erase(w);
  a.erase(q);
  BOOST_CHECK(subset(domain<vec>({x, y}), a));
  BOOST_CHECK(!subset(domain<vec>({x, w}), a));
  BOOST_CHECK(disjoint(domain<vec>({w, q}), a));
  BOOST_CHECK(!disjoint(domain<vec>({w, x}), a));
}
