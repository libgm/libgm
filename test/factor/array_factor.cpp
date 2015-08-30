#define BOOST_TEST_MODULE array_factor
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/base/array_factor.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/assign.hpp>
#include <libgm/functional/eigen.hpp>

namespace libgm {
  template class array_factor<var, 1, double>;
  template class array_factor<var, 2, double>;
}

using namespace libgm;

template <std::size_t N>
struct iarray : public array_factor<var, N, int> {
  typedef array_factor<var, N, int> base;
  typedef typename base::domain_type domain_type;
  typedef typename base::array_type array_type;
  iarray() { }
  explicit iarray(const domain_type& args) { this->reset(args); }
  iarray(const domain_type& args, const array_type& param)
    : base(args, param) { }
  iarray(const domain_type& args, std::initializer_list<int> values)
    : base(args, values) { }
  bool operator==(const iarray& other) const { return this->equal(other); }
  bool operator!=(const iarray& other) const { return !this->equal(other); }
  friend std::ostream&
  operator<<(std::ostream& out, const iarray& f) {
    out << f.arguments() << std::endl
        << f.param() << std::endl;
    return out;
  }
};

typedef iarray<1> iarray1;
typedef iarray<2> iarray2;
typedef array_domain<var, 1> unary_domain;

BOOST_AUTO_TEST_CASE(test_join) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  iarray1 fx({x}, {1, 2});
  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray1 gx({x}, {0, 1});
  iarray1 gy({y}, {1, 2, 3});
  iarray2 gxy({x, y}, {0, 1, 2, 3, 4, 5});
  iarray2 gyx({y, x}, gxy.param().transpose());

  iarray1 h1;
  iarray2 h2;
  h1 = join<iarray1>(fx, gx, libgm::plus<>());
  BOOST_CHECK_EQUAL(h1, iarray1({x}, {1, 3}));

  //h = join<iarray>(fx, gy, libgm::plus<>());
  //BOOST_CHECK_EQUAL(h, iarray({x, y}, {2, 3, 3, 4, 4, 5}));

  h2 = join<iarray2>(fxy, gx, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -3, 5, 5, 7}));

  h2 = join<iarray2>(fxy, gy, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {0, -1, -1, 6, 8, 9}));

  h2 = join<iarray2>(gx, fxy, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -3, 5, 5, 7}));

  h2 = join<iarray2>(gy, fxy, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({y, x}, {0, -1, 8, -1, 6, 9}));

  h2 = join<iarray2>(fxy, gxy, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));

  h2 = join<iarray2>(fxy, gyx, libgm::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));
}

BOOST_AUTO_TEST_CASE(test_join_inplace) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  iarray1 fx({x}, {1, 2});
  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray1 gx({x}, {0, 1});
  iarray2 gxy({x, y}, {0, 1, 2, 3, 4, 5});

  iarray1 hx;
  join_inplace(hx = gx, fx, libgm::plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {1, 3}));

  iarray2 hxy;
  join_inplace(hxy = gxy, fx, libgm::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {1, 3, 3, 5, 5, 7}));

  join_inplace(hxy = gxy, fy, libgm::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {2, 3, 5, 6, 8, 9}));

  join_inplace(hxy = gxy, fxy, libgm::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));

  join_inplace(hxy = gxy, fyx, libgm::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {-1, 5, 0, 8, 1, 11}));
}

BOOST_AUTO_TEST_CASE(test_aggregate) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});

  iarray1 hx;
  aggregate(fxy, unary_domain{x}, hx, sum_op());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {1, 8}));

  iarray1 hy;
  aggregate(fxy, unary_domain{y}, hy, sum_op());
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-3, 1, 11}));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);
  var z = var::discrete(u, "z", 3);
  uint_assignment<var> empty = {{z, 2}};

  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});

  iarray1 hy;
  restrict_assign(fxy, uint_assignment<var>{{x, 1}}, hy);
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-2, 4, 6}));

  iarray1 hx;
  restrict_assign(fxy, uint_assignment<var>{{y, 2}}, hx);
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {5, 6}));
}

BOOST_AUTO_TEST_CASE(test_restrict_join) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);
  var z = var::discrete(u, "z", 2);
  uint_assignment<var> empty = {{z, 1}};

  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fzy({z, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, fxy.param().transpose());
  iarray1 gx({x}, {4, 2});
  iarray1 gy({y}, {1, 2, 0});

  using libgm::plus_assign;
  iarray1 hx;
  hx = gx;
  restrict_join(fxy, uint_assignment<var>{{x, 1}, {y, 2}}, hx, plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {9, 8}));

  hx = gx;
  restrict_join(fxy, uint_assignment<var>{{y, 2}}, hx, plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {9, 8}));

  hx = gx;
  restrict_join(fzy, uint_assignment<var>{{z, 1}, {y, 2}}, hx, plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {10, 8}));

  iarray1 hy = gy;
  restrict_join(fxy, uint_assignment<var>{{x, 1}}, hy, plus_assign<>());
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-1, 6, 6}));
}
