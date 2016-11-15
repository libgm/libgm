#define BOOST_TEST_MODULE canonical_array
#include <boost/test/unit_test.hpp>

#include <libgm/factor/canonical_array.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_table.hpp>
#include <libgm/factor/probability_array.hpp>

#include "predicates.hpp"

namespace libgm {
  template class canonical_array<var, 1>;
  template class canonical_array<var, 2>;
}

using namespace libgm;

typedef canonical_array<var, 1> carray1;
typedef canonical_array<var, 2> carray2;
typedef probability_array<var, 1> parray1;
typedef probability_array<var, 2> parray2;

typedef carray1::param_type param1_type;
typedef carray2::param_type param2_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  carray2 b({x, y});
  BOOST_CHECK(table_properties(b, {x, y}));

  carray1 d({x}, logd(3.0));
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_CLOSE(d[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(d[1], std::log(3.0), 1e-8);

  param2_type params(2, 3);
  params.fill(5.0);
  carray2 f({x, y}, params);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(std::count(f.begin(), f.end(), 5.0), 6);

  carray1 g({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {x}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  carray1 f;
  carray2 g;
  carray2 h;
  g = parray2({x, y}, std::exp(1.0));
  BOOST_CHECK(table_properties(g, {x, y}));
  BOOST_CHECK_CLOSE(g[0], 1.0, 1e-6);

  h.reset({y, x});
  h.param().fill(2.0);
  BOOST_CHECK(table_properties(h, {y, x}));
  BOOST_CHECK_EQUAL(h[0], 2.0);

  swap(g, h);
  BOOST_CHECK(table_properties(g, {y, x}));
  BOOST_CHECK(table_properties(h, {x, y}));
  BOOST_CHECK_EQUAL(g[0], 2.0);
  BOOST_CHECK_EQUAL(h[0], 1.0);

  parray1 pa({x}, {0.5, 0.7});
  f = pa;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.7), 1e-8);

  canonical_table<var> ct({y}, {0.1, 0.2, 0.3});
  f = ct;
  BOOST_CHECK(table_properties(f, {y}));
  BOOST_CHECK_EQUAL(f[0], 0.1);
  BOOST_CHECK_EQUAL(f[1], 0.2);
  BOOST_CHECK_EQUAL(f[2], 0.3);
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  carray2 f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(uint_vector{0,0}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,0}}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,0}}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,1}}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,1}}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,0}, {y,2}}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<var>{{x,1}, {y,2}}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_assignment<var>{{x,0},{y,2}}), 5.0, 1e-8);

  uint_assignment<var> a;
  f.assignment(5, a);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(f.linear_index(a), 5);

  var v = var::discrete(u, "v", 2);
  var w = var::discrete(u, "w", 3);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);
  var z = var::discrete(u, "z", 3);

  carray2 f({x, y}, {0, 1, 2, 3});
  carray1 g({y}, {3, 4});
  carray2 h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + g.log(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2*g.log(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - g.log(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), -g.log(a), 1e-8);
  }

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h *= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 3.0, 1e-8);
  }

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h /= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 1.0, 1e-8);
  }

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - 2.0, 1e-8);
  }

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 - f.log(a), 1e-8);
  }

  h = pow(f, 2.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<var>& a : uint_assignments(domain<var>{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 * f.log(a), 1e-8);
  }

  carray2 f1({x, y}, {0, 1, 2, 3});
  carray2 f2({x, y}, {-2, 3, 0, 0});
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

  carray2 f({x, y}, {0, 1, 2, 3, 5, 6});
  carray1 h;
  uint_assignment<var> a;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.maximum().lv, 6.0);
  BOOST_CHECK_EQUAL(f.maximum(a).lv, 6.0);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);

  h = f.minimum({y});
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.minimum().lv, 0.0);
  BOOST_CHECK_EQUAL(f.minimum(a).lv, 0.0);
  BOOST_CHECK_EQUAL(a[x], 0);
  BOOST_CHECK_EQUAL(a[y], 0);

  parray2 pxy({x, y}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  parray1 py({y}, {1.6, 0.3, 0.4});
  carray2 g(pxy);
  h = g.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(double(g.marginal()), pxy.marginal(), 1e-8);
  BOOST_CHECK_CLOSE(double(h.normalize().marginal()), 1.0, 1e-8);

  h = f.exp_log(py);
  BOOST_CHECK(table_properties(h, {x}));
  BOOST_CHECK_CLOSE(h[0], 0*1.6 + 2*0.3 + 5*0.4, 1e-8);
  BOOST_CHECK_CLOSE(h[1], 1*1.6 + 3*0.3 + 6*0.4, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  carray2 f({x, y}, {0, 1, 2, 3, 5, 6});
  carray1 h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, fr));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);

  parray2 pxy({x, y}, {0.1, 0.2, 0.3, 0.4});
  parray2 qxy({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  carray2 p(pxy);
  carray2 q(qxy);
  carray2 m = (p+q) / logd(2);
  double hpxy = -(0.1*log(0.1) + 0.2*log(0.2) + 0.3*log(0.3) + 0.4*log(0.4));
  double hpx = -(0.4*log(0.4) + 0.6*log(0.6));
  double hpy = -(0.3*log(0.3) + 0.7*log(0.7));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    hpq += -pxy[i] * log(qxy[i]);
    klpq += pxy[i] * log(pxy[i]/qxy[i]);
    double diff = std::abs(std::log(pxy[i]) - std::log(qxy[i]));
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
