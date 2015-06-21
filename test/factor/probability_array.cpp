#define BOOST_TEST_MODULE probability_array
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_array.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_array.hpp>
#include <libgm/factor/probability_table.hpp>

#include "predicates.hpp"

namespace libgm {
  template class probability_array<double, 1, variable>;
  template class probability_array<double, 2, variable>;
}

using namespace libgm;

typedef parray1::param_type param1_type;
typedef parray2::param_type param2_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  parray2 b({x, y});
  BOOST_CHECK(table_properties(b, {x, y}));

  parray1 d({x}, 3.0);
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_CLOSE(d[0], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(d[1], 3.0, 1e-8);

  param2_type params(2, 3);
  params.fill(5.0);
  parray2 f({x, y}, params);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(std::count(f.begin(), f.end(), 5.0), 6);

  parray1 g({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {x}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  parray1 f;
  parray2 g;
  parray2 h;
  g = parray2({x, y}, 1.0);
  BOOST_CHECK(table_properties(g, {x, y}));
  BOOST_CHECK_EQUAL(g[0], 1.0);

  h.reset({y, x});
  h.param().fill(2.0);
  BOOST_CHECK(table_properties(h, {y, x}));
  BOOST_CHECK_EQUAL(h[0], 2.0);

  swap(g, h);
  BOOST_CHECK(table_properties(g, {y, x}));
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK_EQUAL(g[0], 2.0);
  BOOST_CHECK_EQUAL(h[0], 1.0);

  carray1 ca({x}, {0.5, 0.7});
  f = ca;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::exp(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::exp(0.7), 1e-8);

  ptable ct({y}, {0.1, 0.2, 0.3});
  f = ct;
  BOOST_CHECK(table_properties(f, {y}));
  BOOST_CHECK_EQUAL(f[0], 0.1);
  BOOST_CHECK_EQUAL(f[1], 0.2);
  BOOST_CHECK_EQUAL(f[2], 0.3);
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  parray2 f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(uint_vector{0,0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,0}}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,0}}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,1}}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,1}}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,2}}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,2}}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), std::log(5.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_assignment<>{{x,0},{y,2}}), std::log(5.0), 1e-8);

  uint_assignment<> a;
  f.assignment(5, a);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(f.linear_index(a), 5);

  variable v = u.new_discrete_variable("v", 2);
  variable w = u.new_discrete_variable("w", 3);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 2);

  parray2 f({x, y}, {0, 1, 2, 3});
  parray1 g({y}, {3, 4});
  parray2 h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a) * g(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / g(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) ? (1.0 / g(a)) : 0.0, 1e-8);
  }

  h = f * 2.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h *= 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 6.0, 1e-8);
  }

  h = 2.0 * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h /= 4.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 0.5, 1e-8);
  }

  h = f / 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / 3.0, 1e-8);
  }

  h = 3.0 / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    if (f(a)) {
      BOOST_CHECK_CLOSE(h(a), 3.0 / f(a), 1e-8);
    } else {
      BOOST_CHECK(std::isinf(h(a)));
    }
  }

  h = pow(f, 3.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h(a), std::pow(f(a), 3.0), 1e-8);
  }

  parray2 f1({x, y}, {0, 1, 2, 3});
  parray2 f2({x, y}, {-2, 3, 0, 0});
  std::vector<double> fmax = {0, 3, 2, 3};
  std::vector<double> fmin = {-2, 1, 0, 0};

  h = max(f1, f2);
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK(range_equal(h, fmax));

  h = min(f1, f2);
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK(range_equal(h, fmin));

  h = weighted_update(f1, f2, 0.3);
  for (std::size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h[i], 0.7 * f1[i] + 0.3 * f2[i], 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_collapse) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  parray2 f({x, y}, {0, 1, 2, 3, 5, 6});
  parray1 h;
  uint_assignment<> a;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.maximum(), 6.0);
  BOOST_CHECK_EQUAL(f.maximum(a), 6.0);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);

  h = f.minimum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.minimum(), 0.0);
  BOOST_CHECK_EQUAL(f.minimum(a), 0.0);
  BOOST_CHECK_EQUAL(a[x], 0);
  BOOST_CHECK_EQUAL(a[y], 0);

  parray2 pxy({x, y}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  parray1 py({y}, {1.6, 0.3, 0.4});
  h = pxy.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(h[i], py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(pxy.marginal(), 1.1+0.5+0.1+0.2+0.4, 1e-8);
  BOOST_CHECK_CLOSE(h.normalize().marginal(), 1.0, 1e-8);

  parray1 qx({x}, {0.4, 0.6});
  parray1 qy({y}, {0.2, 0.5, 0.3});

  parray1 rx = pxy.product_marginal(qy, {x});
  BOOST_CHECK(table_properties(rx, {x}));
  BOOST_CHECK_CLOSE(rx[0], 1.1*0.2+0.1*0.5+0.4*0.3, 1e-8);
  BOOST_CHECK_CLOSE(rx[1], 0.5*0.2+0.2*0.5+0.0*0.3, 1e-8);

  parray1 ry = qx.product_marginal(pxy, {y});
  BOOST_CHECK(table_properties(ry, {y}));
  BOOST_CHECK_CLOSE(ry[0], 1.1*0.4 + 0.5*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[1], 0.1*0.4 + 0.2*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[2], 0.4*0.4 + 0.0*0.6, 1e-8);

  parray1 sx = pxy.product_marginal(qx, {x});
  BOOST_CHECK(table_properties(sx, {x}));
  BOOST_CHECK_CLOSE(sx[0], (1.1+0.1+0.4)*0.4, 1e-8);
  BOOST_CHECK_CLOSE(sx[1], (0.5+0.2+0.0)*0.6, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  parray2 f({x, y}, {0, 1, 2, 3, 5, 6});
  parray1 h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, fr));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 2);

  parray2 p({x, y}, {0.1, 0.2, 0.3, 0.4});
  parray2 q({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  parray2 m = (p+q) / 2.0;
  double hpxy = -(0.1*log(0.1) + 0.2*log(0.2) + 0.3*log(0.3) + 0.4*log(0.4));
  double hpx = -(0.4*log(0.4) + 0.6*log(0.6));
  double hpy = -(0.3*log(0.3) + 0.7*log(0.7));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    hpq += -p[i] * log(q[i]);
    klpq += p[i] * log(p[i]/q[i]);
    double diff = std::abs(p[i] - q[i]);
    sumdiff += diff;
    maxdiff = std::max(maxdiff, diff);
  }
  double jspq = (kl_divergence(p, m) + kl_divergence(q, m)) / 2;
  BOOST_CHECK_CLOSE(p.entropy(), hpxy, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy({x}), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy({y}), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information({x}, {y}), klpq, 1e-6);
  BOOST_CHECK_CLOSE(cross_entropy(p, q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(kl_divergence(p, q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(js_divergence(p, q), jspq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}
