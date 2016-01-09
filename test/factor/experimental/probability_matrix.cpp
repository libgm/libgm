#define BOOST_TEST_MODULE probability_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/probability_matrix.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/experimental/logarithmic_matrix.hpp>
#include <libgm/factor/experimental/probability_table.hpp>

#include "../predicates.hpp"

namespace libgm { namespace experimental {
  template class probability_matrix<var>;
} }

using namespace libgm;

typedef experimental::logarithmic_vector<var> lvector;
typedef experimental::logarithmic_matrix<var> lmatrix;
typedef experimental::probability_vector<var> pvector;
typedef experimental::probability_matrix<var> pmatrix;
typedef experimental::probability_table<var> ptable;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  pmatrix a({x, y});
  BOOST_CHECK(table_properties(a, {x, y}));

  pmatrix b({x, y}, 3.0);
  BOOST_CHECK(table_properties(b, {x, y}));
  BOOST_CHECK_CLOSE(b(0, 0), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(b(1, 2), 3.0, 1e-8);

  pmatrix c({x, y}, real_matrix<>::Constant(2, 3, 5.0));
  BOOST_CHECK(table_properties(c, {x, y}));
  BOOST_CHECK_EQUAL(std::count(c.begin(), c.end(), 5.0), 6);

  pmatrix d({x, y}, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(table_properties(d, {x, y}));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(d[i], i + 1.0, 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  pmatrix g;
  pmatrix h;
  g = pmatrix({x, y}, 1.0);
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

  pmatrix f = lmatrix({x, y}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).probability();
  BOOST_CHECK(table_properties(f, {x, y}));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], std::exp((i+1) * 0.1), 1e-8);
  }

  f = ptable({y, x}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(table_properties(f, {y, x}));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], (6-i) * 0.1, 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  pmatrix f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(uint_vector{0,0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,0}}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,0}}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,1}}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,1}}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,2}}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,2}}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(0, 0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 0), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0, 1), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 1), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0, 2), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 2), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), std::log(5.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_assignment<var>{{x,0},{y,2}}), std::log(5.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(0, 2), std::log(5.0), 1e-8);

  uint_assignment<var> a;
  a.insert_or_assign(f.arguments(), {1, 2});
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(a.linear_index(f.arguments()), 5);

  var v = var::discrete(u, "v", 2);
  var w = var::discrete(u, "w", 3);
  f.subst_args(std::unordered_map<var, var>({{x, v}, {y, w}}));
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);

  pmatrix f({x, y}, {0, 1, 2, 3});
  pvector g({y}, {3, 4});
  pmatrix h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * g(a) * g(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / g(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) ? (1.0 / g(a)) : 0.0, 1e-8);
  }

  h = f * 2.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h *= 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 6.0, 1e-8);
  }

  h = 2.0 * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 2.0, 1e-8);
  }

  h /= 4.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) * 0.5, 1e-8);
  }

  h = f / 3.0;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), f(a) / 3.0, 1e-8);
  }

  h = 3.0 / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    if (f(a)) {
      BOOST_CHECK_CLOSE(h(a), 3.0 / f(a), 1e-8);
    } else {
      BOOST_CHECK(std::isinf(h(a)));
    }
  }

  h = pow(f, 3.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h(a), std::pow(f(a), 3.0), 1e-8);
  }

  pmatrix f1({x, y}, {0, 1, 2, 3});
  pmatrix f2({x, y}, {-2, 3, 0, 0});
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
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  pmatrix f({x, y}, {0, 1, 2, 3, 5, 6});
  pvector h;
  uint_assignment<var> a;

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

  pmatrix pxy({x, y}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  pvector py({y}, {1.6, 0.3, 0.4});
  h = pxy.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(h[i], py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(pxy.marginal(), 1.1+0.5+0.1+0.2+0.4, 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(h.marginal(), 1.0, 1e-8);

  pvector qx({x}, {0.4, 0.6});
  pvector qy({y}, {0.2, 0.5, 0.3});

  pvector rx = (pxy * qy).marginal(x);
  BOOST_CHECK(table_properties(rx, {x}));
  BOOST_CHECK_CLOSE(rx[0], 1.1*0.2+0.1*0.5+0.4*0.3, 1e-8);
  BOOST_CHECK_CLOSE(rx[1], 0.5*0.2+0.2*0.5+0.0*0.3, 1e-8);

  pvector ry = (qx * pxy).marginal(y);
  BOOST_CHECK(table_properties(ry, {y}));
  BOOST_CHECK_CLOSE(ry[0], 1.1*0.4 + 0.5*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[1], 0.1*0.4 + 0.2*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[2], 0.4*0.4 + 0.0*0.6, 1e-8);

  pvector sx = (pxy * qx).marginal(x);
  BOOST_CHECK(table_properties(sx, {x}));
  BOOST_CHECK_CLOSE(sx[0], (1.1+0.1+0.4)*0.4, 1e-8);
  BOOST_CHECK_CLOSE(sx[1], (0.5+0.2+0.0)*0.6, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  pmatrix f({x, y}, {0, 1, 2, 3, 5, 6});
  pvector h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, fr));
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);

  pmatrix p({x, y}, {0.1, 0.2, 0.3, 0.4});
  pmatrix q({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  pmatrix m = (p+q) / 2.0;
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
