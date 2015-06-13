#define BOOST_TEST_MODULE basic_domain
#include <boost/test/unit_test.hpp>

#include <libgm/argument/basic_domain.hpp>

#include <libgm/argument/universe.hpp>

namespace libgm {
  template class basic_domain<variable>;
  template class basic_domain<dprocess>;
  template class basic_domain<std::size_t>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  domain a;
  BOOST_CHECK(a.empty());

  domain b({x, y});
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], x);
  BOOST_CHECK_EQUAL(b[1], y);

  domain c(std::vector<variable>(1, x));
  BOOST_CHECK_EQUAL(c.size(), 1);
  BOOST_CHECK_EQUAL(c[0], x);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);
  variable z = u.new_discrete_variable("z", 4);
  variable w = u.new_discrete_variable("w", 3);

  domain xyz  = {x, y, z};
  domain x1   = {x};
  domain y1   = {y};
  domain z1   = {z};
  domain xy   = {x, y};
  domain xw   = {x, w};
  domain yx   = {y, x};
  domain yw   = {y, w};
  domain yz   = {y, z};
  domain zw   = {z, w};
  domain xyw  = {x, y, w};
  domain yzw  = {y, z, w};
  domain xyzw = {x, y, z, w};
  domain xwzy = {x, w, z, y};
  domain xywx = {x, y, w, x};

  BOOST_CHECK_EQUAL(x1 + y1, xy);
  BOOST_CHECK_EQUAL(xy + z1, xyz);
  BOOST_CHECK_EQUAL(xy - z1, xy);
  BOOST_CHECK_EQUAL(xy - yz, x1);
  BOOST_CHECK_EQUAL(xy | z1, xyz);
  BOOST_CHECK_EQUAL(xy | yw, xyw);
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

  variable_map map;
  map[x] = x;
  map[y] = w;
  map[z] = z;
  map[w] = y;
  xyzw.subst(map);
  BOOST_CHECK_EQUAL(xyzw, xwzy);
}

BOOST_AUTO_TEST_CASE(test_num_assignments) {
  universe u;

  domain v;
  v.push_back(u.new_discrete_variable("a", 2));
  v.push_back(u.new_discrete_variable("b", 3));
  v.push_back(u.new_discrete_variable("c", 4));
  v.push_back(u.new_discrete_variable("d", 5));
  v.push_back(u.new_discrete_variable("e", 6));
  v.push_back(u.new_discrete_variable("f", 7));
  v.push_back(u.new_discrete_variable("g", 8));
  v.push_back(u.new_discrete_variable("h", 9));
  v.push_back(u.new_discrete_variable("i", 10));
  v.push_back(u.new_discrete_variable("j", 11));

  BOOST_CHECK_EQUAL(num_values(v), 39916800);

  v.push_back(u.new_discrete_variable("k", 1000000));
  v.push_back(u.new_discrete_variable("l", 1000000));
  v.push_back(u.new_discrete_variable("m", 1000000));
  v.push_back(u.new_discrete_variable("n", 1000000));
  BOOST_CHECK_THROW(num_values(v), std::out_of_range);
}
