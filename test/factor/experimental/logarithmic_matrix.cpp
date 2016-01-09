#define BOOST_TEST_MODULE logarithmic_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/logarithmic_matrix.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/experimental/logarithmic_table.hpp>
#include <libgm/factor/experimental/probability_matrix.hpp>

#include "../predicates.hpp"

namespace libgm { namespace experimental {
  template class logarithmic_matrix<var, double>;
} }

using namespace libgm;

typedef experimental::logarithmic_vector<var> lvector;
typedef experimental::logarithmic_matrix<var> lmatrix;
typedef experimental::probability_vector<var> pvector;
typedef experimental::probability_matrix<var> pmatrix;
typedef experimental::logarithmic_table<var> ltable;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lmatrix a({x, y});
  BOOST_CHECK(table_properties(a, {x, y}));

  lmatrix b({x, y}, logd(3.0));
  BOOST_CHECK(table_properties(b, {x, y}));
  BOOST_CHECK_CLOSE(b(0, 0).lv, std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(b(1, 2).lv, std::log(3.0), 1e-8);

  lmatrix c({x, y}, real_matrix<>::Constant(2, 3, 5.0));
  BOOST_CHECK(table_properties(c, {x, y}));
  BOOST_CHECK_EQUAL(std::count(c.begin(), c.end(), 5.0), 6);

  lmatrix d({x, y}, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(table_properties(d, {x, y}));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(d[i], i + 1.0, 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lmatrix f;
  lmatrix g;
  lmatrix h;
  g = pmatrix({x, y}, std::exp(1.0)).logarithmic();
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
  BOOST_CHECK_CLOSE(h[0], 1.0, 1e-8);

  f = ltable({y, x}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(table_properties(f, {y, x}));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], (6-i) * 0.1, 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lmatrix f({x, y});
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
  a.insert_or_assign(f.arguments(), 5);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(a.linear_index(f.arguments()), 5);

  var v = var::discrete(u, "v", 2);
  var w = var::discrete(u, "w", 3);
  f.subst_args(std::unordered_map<var, var>{{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);
  var z = var::discrete(u, "z", 3);

  lmatrix f({x, y}, {0, 1, 2, 3});
  lvector g({y}, {3, 4});
  lmatrix h;
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

  lmatrix f1({x, y}, {0, 1, 2, 3});
  lmatrix f2({x, y}, {-2, 3, 0, 0});
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

  lmatrix f({x, y}, {0, 1, 2, 3, 5, 6});
  lvector h;
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

  pmatrix pxy({x, y}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  pvector py({y}, {1.6, 0.3, 0.4});
  lmatrix g = pxy.logarithmic();
  h = g.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(double(g.marginal()), pxy.marginal(), 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(double(h.marginal()), 1.0, 1e-8);

  /*
  h = f.exp_log(py);
  BOOST_CHECK(table_properties(h, {x}));
  BOOST_CHECK_CLOSE(h[0], 0*1.6 + 2*0.3 + 5*0.4, 1e-8);
  BOOST_CHECK_CLOSE(h[1], 1*1.6 + 3*0.3 + 6*0.4, 1e-8);
  */
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 3);

  lmatrix f({x, y}, {0, 1, 2, 3, 5, 6});
  lvector h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, fr));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 2);

  pmatrix pxy({x, y}, {0.1, 0.2, 0.3, 0.4});
  pmatrix qxy({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  lmatrix p = pxy.logarithmic();
  lmatrix q = qxy.logarithmic();
  lmatrix m = (p+q) / logd(2);
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
