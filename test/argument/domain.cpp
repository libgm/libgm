#define BOOST_TEST_MODULE domain
#include <boost/test/unit_test.hpp>

#include <libgm/argument/domain.hpp>
#include <libgm/argument/sequence.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/datastructure/uint_vector.hpp>

namespace libgm {
  template class domain<var>;
  template class domain<vec>;
  template class domain<sequence<var> >;
  template class domain<sequence<vec> >;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  domain<var> a;
  BOOST_CHECK(a.empty());

  domain<var> b({x, y});
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], x);
  BOOST_CHECK_EQUAL(b[1], y);

  domain<var> c(std::vector<var>(1, x));
  BOOST_CHECK_EQUAL(c.size(), 1);
  BOOST_CHECK_EQUAL(c[0], x);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);
  var z = var::discrete(u, "z", 4);
  var w = var::discrete(u, "w", 3);

  domain<var> xyz  = {x, y, z};
  domain<var> x1   = {x};
  domain<var> y1   = {y};
  domain<var> z1   = {z};
  domain<var> xy   = {x, y};
  domain<var> xw   = {x, w};
  domain<var> yx   = {y, x};
  domain<var> yw   = {y, w};
  domain<var> yz   = {y, z};
  domain<var> zw   = {z, w};
  domain<var> xyw  = {x, y, w};
  domain<var> yzw  = {y, z, w};
  domain<var> xyzw = {x, y, z, w};
  domain<var> xwzy = {x, w, z, y};
  domain<var> xywx = {x, y, w, x};

  BOOST_CHECK_EQUAL(concat(x1, y1), xy);
  BOOST_CHECK_EQUAL(concat(xy, z1), xyz);
  BOOST_CHECK(xyzw.prefix(xy));
  BOOST_CHECK(!xyzw.prefix(xw));
  BOOST_CHECK(xyzw.suffix(zw));
  BOOST_CHECK(!xyzw.suffix(xw));

  domain<var> present, absent;
  xyw.partition(yzw, present, absent);
  BOOST_CHECK_EQUAL(present, yw);
  BOOST_CHECK_EQUAL(absent, x1);

  BOOST_CHECK_EQUAL(xy + z1, xyz);
  BOOST_CHECK_EQUAL(xy + yw, xyw);
  BOOST_CHECK_EQUAL(xy - z1, xy);
  BOOST_CHECK_EQUAL(xy - yz, x1);
  BOOST_CHECK_EQUAL(xy & yw, y1);

  BOOST_CHECK(disjoint(xy, z1));
  BOOST_CHECK(!disjoint(xy, yzw));
  BOOST_CHECK(equivalent(xy, yx));
  BOOST_CHECK(!equivalent(yw, zw));
  BOOST_CHECK(subset(yx, xyz));
  BOOST_CHECK(!subset(yx, yw));
  BOOST_CHECK(superset(xyzw, yx));
  BOOST_CHECK(!superset(xyw, xyz));
  BOOST_CHECK(compatible(xy, xw));
  BOOST_CHECK(!compatible(xyz, xyw));
  BOOST_CHECK(!compatible(x1, y1));

  BOOST_CHECK_EQUAL(xyz.count(x), 1);
  BOOST_CHECK_EQUAL(xyz.count(w), 0);
  xywx.unique();
  BOOST_CHECK_EQUAL(xywx.size(), 3);
  BOOST_CHECK(equivalent(xywx, xyw));

  std::unordered_map<var, var> map;
  map[x] = x;
  map[y] = w;
  map[z] = z;
  map[w] = y;
  xyzw.substitute(map);
  BOOST_CHECK_EQUAL(xyzw, xwzy);
}

BOOST_AUTO_TEST_CASE(test_num_univariate) {
  universe u;

  domain<var> v;
  v.push_back(var::discrete(u, "a", 2));
  v.push_back(var::discrete(u, "b", 3));
  v.push_back(var::discrete(u, "c", 4));
  v.push_back(var::discrete(u, "d", 5));
  v.push_back(var::discrete(u, "e", 6));
  v.push_back(var::discrete(u, "f", 7));
  v.push_back(var::discrete(u, "g", 8));
  v.push_back(var::discrete(u, "h", 9));
  v.push_back(var::discrete(u, "i", 10));
  v.push_back(var::discrete(u, "j", 11));

  BOOST_CHECK_EQUAL(v.num_values(), 39916800);
  BOOST_CHECK_EQUAL(v.num_dimensions(), 10);

  v.clear();
  v.push_back(var::discrete(u, "k", 1000000));
  v.push_back(var::discrete(u, "l", 1000000));
  v.push_back(var::discrete(u, "m", 1000000));
  v.push_back(var::discrete(u, "n", 1000000));
  BOOST_CHECK_THROW(v.num_values(), std::out_of_range);
  BOOST_CHECK_EQUAL(v.num_dimensions(), 4);
}

BOOST_AUTO_TEST_CASE(test_num_multivariate) {
  universe u;

  domain<vec> v;
  v.push_back(vec::discrete(u, "a", {2, 3, 4}));
  v.push_back(vec::discrete(u, "d", {5, 6}));
  v.push_back(vec::discrete(u, "f", {7, 8, 9, 10}));
  v.push_back(vec::discrete(u, "j", 11));

  BOOST_CHECK_EQUAL(v.num_values(), 39916800);
  BOOST_CHECK_EQUAL(v.num_dimensions(), 10);

  v.clear();
  v.push_back(vec::discrete(u, "k", {1000000, 1000000}));
  v.push_back(vec::discrete(u, "m", {1000000, 1000000}));
  BOOST_CHECK_THROW(v.num_values(), std::out_of_range);
  BOOST_CHECK_EQUAL(v.num_dimensions(), 4);
}
