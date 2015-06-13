#define BOOST_TEST_MODULE typesafe_union
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/typesafe_union.hpp>

namespace libgm {
  template class typesafe_union<double, int, char>;
}

typedef libgm::typesafe_union<double, const char*> union_type;

BOOST_AUTO_TEST_CASE(test_assign) {
  union_type u;
  BOOST_CHECK(u.empty());

  u = 1.2;
  BOOST_CHECK_EQUAL(u.which(), 0);
  BOOST_CHECK_EQUAL(u.get<0>(), 1.2);
  BOOST_CHECK_EQUAL(u.get<double>(), 1.2);

  u = "hello world";
  BOOST_CHECK_EQUAL(u.which(), 1);
  BOOST_CHECK_EQUAL(u.get<1>(), std::string("hello world"));
  BOOST_CHECK_EQUAL(u.get<const char*>(), std::string("hello world"));

  union_type v = u;
  BOOST_CHECK_EQUAL(v, u);

  u = union_type();
  BOOST_CHECK_NE(v, u);
  BOOST_CHECK_EQUAL(v.which(), 1);
  BOOST_CHECK_EQUAL(v.get<1>(), std::string("hello world"));
  BOOST_CHECK_EQUAL(v.get<const char*>(), std::string("hello world"));
}

BOOST_AUTO_TEST_CASE(test_swap) {
  union_type u = 1.5;
  union_type v = "a";
  swap(u, v);
  BOOST_CHECK_EQUAL(u.which(), 1);
  BOOST_CHECK_EQUAL(v.which(), 0);
  BOOST_CHECK_EQUAL(u.get<1>(), std::string("a"));
  BOOST_CHECK_EQUAL(v.get<0>(), 1.5);
}

BOOST_AUTO_TEST_CASE(test_compare) {
  union_type w;
  union_type x = 1.5;
  union_type y = 1.8;
  union_type z = "yup";

  BOOST_CHECK(w == w);
  BOOST_CHECK(x == x);
  BOOST_CHECK(z == z);
  BOOST_CHECK(!(x == w));

  BOOST_CHECK(x != w);
  BOOST_CHECK(x != y);
  BOOST_CHECK(x != z);
  BOOST_CHECK(!(w != w));

  BOOST_CHECK(x < y);
  BOOST_CHECK(x < z);
  BOOST_CHECK(w < x);
  BOOST_CHECK(!(w < w));
  BOOST_CHECK(!(x < x));

  BOOST_CHECK(x <= x);
  BOOST_CHECK(x <= y);
  BOOST_CHECK(x <= z);
  BOOST_CHECK(!(x <= w));

  BOOST_CHECK(x > w);
  BOOST_CHECK(y > x);
  BOOST_CHECK(z > y);
  BOOST_CHECK(!(w > w));
  BOOST_CHECK(!(x > x));

  BOOST_CHECK(w >= w);
  BOOST_CHECK(x >= w);
  BOOST_CHECK(y >= x);
  BOOST_CHECK(z >= y);
}
