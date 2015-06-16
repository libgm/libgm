#define BOOST_TEST_MODULE canonical_table
#include <boost/test/unit_test.hpp>

#include <libgm/argument/uint_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_table.hpp>
#include <libgm/factor/probability_table.hpp>

#include "predicates.hpp"

namespace libgm {
  template class canonical_table<double, variable>;
  template class canonical_table<float, variable>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector)

typedef ctable::param_type param_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  ctable a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.arguments().empty());

  ctable b({x, y});
  BOOST_CHECK(table_properties(b, {x, y}));

  ctable c(logd(2.0));
  BOOST_CHECK(table_properties(c, {}));
  BOOST_CHECK_CLOSE(c[0], std::log(2.0), 1e-8);

  ctable d({x}, logd(3.0));
  BOOST_CHECK(table_properties(d, {x}));
  BOOST_CHECK_CLOSE(d[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(d[1], std::log(3.0), 1e-8);

  param_type params({2, 3}, 5.0);
  ctable f({x, y}, params);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK_EQUAL(std::count(f.begin(), f.end(), 5.0), 6);

  ctable g({x}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {x}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  ctable f;
  f = logd(2.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(2.0), 1e-8);

  f.reset({x, y});
  BOOST_CHECK(table_properties(f, {x, y}));

  f = logd(3.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(3.0), 1e-8);

  ptable pt({x}, {0.5, 0.7});
  f = pt;
  BOOST_CHECK(table_properties(f, {x}));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.7), 1e-8);

  ctable g({x, y});
  swap(f, g);
  BOOST_CHECK(table_properties(f, {x, y}));
  BOOST_CHECK(table_properties(g, {x}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  ctable f({x, y});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f(uint_vector{0,0}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,0}}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,0}}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,1}}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,1}}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,0}, {y,2}}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_assignment<>{{x,1}, {y,2}}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_assignment<>{{x,0},{y,2}}), 5.0, 1e-8);

  uint_assignment<> a;
  f.assignment({1, 2}, a);
  BOOST_CHECK_EQUAL(a[x], 1);
  BOOST_CHECK_EQUAL(a[y], 2);
  BOOST_CHECK_EQUAL(f.index(a), 5);

  variable v = u.new_discrete_variable("v", 2);
  variable w = u.new_discrete_variable("w", 3);
  f.subst_args({{x, v}, {y, w}});
  BOOST_CHECK(table_properties(f, {v, w}));
}


BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 2);
  variable z = u.new_discrete_variable("z", 3);

  ctable f({x, y}, {0, 1, 2, 3});
  ctable g({y, z}, {1, 2, 3, 4, 5, 6});
  ctable h;
  h = f * g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + g.log(a), 1e-8);
  }

  h *= g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2*g.log(a), 1e-8);
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - g.log(a), 1e-8);
  }

  h /= f;
  BOOST_CHECK(table_properties(h, {x, y, z}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y, z})) {
    BOOST_CHECK_CLOSE(h.log(a), -g.log(a), 1e-8);
  }

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h *= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 3.0, 1e-8);
  }

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 2.0, 1e-8);
  }

  h /= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) + 1.0, 1e-8);
  }

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), f.log(a) - 2.0, 1e-8);
  }

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 - f.log(a), 1e-8);
  }

  h = pow(f, 2.0);
  BOOST_CHECK(table_properties(h, {x, y}));
  for (const uint_assignment<>& a : uint_assignments(domain{x, y})) {
    BOOST_CHECK_CLOSE(h.log(a), 2.0 * f.log(a), 1e-8);
  }

  ctable f1({x, y}, {0, 1, 2, 3});
  ctable f2({x, y}, {-2, 3, 0, 0});
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

  ctable f({x, y}, {0, 1, 2, 3, 5, 6});
  ctable h;
  uint_assignment<> a;

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

  double pxy[] = {1.1, 0.5, 0.1, 0.2, 0.4, 0.0};
  double py[] = {1.6, 0.3, 0.4};
  ctable g({x, y});
  std::transform(pxy, pxy + 6, g.begin(), logarithm<double>());
  h = g.marginal({y});
  BOOST_CHECK(table_properties(h, {y}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(double(g.marginal()), std::accumulate(pxy, pxy + 6, 0.0), 1e-8);
  BOOST_CHECK_CLOSE(double(h.normalize().marginal()), 1.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);

  ctable f({x, y}, {0, 1, 2, 3, 5, 6});
  ctable h = f.restrict({{x, 1}});
  std::vector<double> fr = {1, 3, 6};
  BOOST_CHECK(table_properties(h, {y}));
  BOOST_CHECK(range_equal(h, fr));
}


BOOST_AUTO_TEST_CASE(test_sample) {
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 3);
  ctable f({x, y}, {0, 0.1, 0.2, 0.3, 0.5, 0.6});
  f.normalize();
  std::mt19937 rng1;
  std::mt19937 rng2;
  std::mt19937 rng3;
  uint_assignment<> a;

  // test marginal sample
  auto fd = f.distribution();
  for (std::size_t i = 0; i < 20; ++i) {
    uint_vector sample = fd(rng1);
    BOOST_CHECK_EQUAL(f.sample(rng2), sample);
    f.sample(rng3, a);
    BOOST_CHECK_EQUAL(a[x], sample[0]);
    BOOST_CHECK_EQUAL(a[y], sample[1]);
  }

  // test conditional sample
  ctable g = f.conditional({y});
  auto gd = g.distribution();
  for (std::size_t yv = 0; yv < 3; ++yv) {
    uint_vector tail(1, yv);
    a[y] = yv;
    for (std::size_t i = 0; i < 20; ++i) {
      uint_vector sample = gd(rng1, tail);
      BOOST_CHECK_EQUAL(g.sample(rng2, tail), sample);
      g.sample(rng3, {x}, a);
      BOOST_CHECK_EQUAL(a[x], sample[0]);
      BOOST_CHECK_EQUAL(a[y], yv);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  universe u;
  variable x = u.new_discrete_variable("x", 2);
  variable y = u.new_discrete_variable("y", 2);

  ptable pxy({x, y}, {0.1, 0.2, 0.3, 0.4});
  ptable qxy({x, y}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  ctable p(pxy);
  ctable q(qxy);
  ctable m = (p+q) / logd(2);
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
