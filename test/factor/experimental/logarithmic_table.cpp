#define BOOST_TEST_MODULE logarithmic_table
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/logarithmic_table.hpp>

#include <libgm/iterator/uint_vector_iterator.hpp>
#include <libgm/factor/experimental/probability_table.hpp>
#include <libgm/factor/experimental/logarithmic_vector.hpp>
#include <libgm/factor/experimental/logarithmic_matrix.hpp>

#include "predicates.hpp"

namespace libgm { namespace experimental {
  template class logarithmic_table<double>;
  template class logarithmic_table<float>;
} }

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector)

using ltable = experimental::logarithmic_table<>;
using ptable = experimental::probability_table<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  ltable a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.arity() == 0);

  ltable b({2, 3});
  BOOST_CHECK(table_properties(b, {2, 3}));

  ltable c(logd(2.0));
  BOOST_CHECK(table_properties(c, {}));
  BOOST_CHECK_CLOSE(c[0], std::log(2.0), 1e-8);

  ltable d({2}, logd(3.0));
  BOOST_CHECK(table_properties(d, {2}));
  BOOST_CHECK_CLOSE(d[0], std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(d[1], std::log(3.0), 1e-8);

  table<double> params({2, 3}, 5.0);
  ltable f(params);
  BOOST_CHECK(table_properties(f, {2, 3}));
  BOOST_CHECK_EQUAL(std::count(f.begin(), f.end(), 5.0), 6);

  ltable g({2}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {2}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  ltable f;
  f = logd(2.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(2.0), 1e-8);

  f.reset({2, 3});
  BOOST_CHECK(table_properties(f, {2, 3}));

  f = logd(3.0);
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], std::log(3.0), 1e-8);

  ptable pt({2}, {0.5, 0.7});
  f = pt.logarithmic();
  BOOST_CHECK(table_properties(f, {2}));
  BOOST_CHECK_CLOSE(f[0], std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f[1], std::log(0.7), 1e-8);

  ltable g({2, 3});
  swap(f, g);
  BOOST_CHECK(table_properties(f, {2, 3}));
  BOOST_CHECK(table_properties(g, {2}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  ltable f({2, 3});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f({0,0}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,0}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0,1}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,1}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0,2}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,2}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log({0,2}), 5.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_operators) {
  ltable f({2, 2}, {0, 1, 2, 3});       // x, y
  ltable g({2, 3}, {1, 2, 3, 4, 5, 6}); // y, z
  ltable h;
  h = f.dim(1) * g.dim(0);
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log({v[0],v[1]}) + g.log({v[1],v[2]}), 1e-8);
  }

  h.dims({1, 2}) *= g;
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log({v[0],v[1]}) + 2*g.log({v[1],v[2]}), 1e-8);
  }

  h = f.tail(1) / g.head(1);
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log({v[0],v[1]}) - g.log({v[1],v[2]}), 1e-8);
  }

  h.dims({0, 1}) /= f;
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), -g.log({v[1],v[2]}), 1e-8);
  }

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 2.0, 1e-8);
  }

  h *= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 3.0, 1e-8);
  }

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 2.0, 1e-8);
  }

  h /= logd(1.0, log_tag());
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 1.0, 1e-8);
  }

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) - 2.0, 1e-8);
  }

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), 2.0 - f.log(v), 1e-8);
  }

  h = pow(f, 2.0);
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h.log(v), 2.0 * f.log(v), 1e-8);
  }

  ltable f1({2, 2}, {0, 1, 2, 3});
  ltable f2({2, 2}, {-2, 3, 0, 0});
  std::vector<double> fmax = {0, 3, 2, 3};
  std::vector<double> fmin = {-2, 1, 0, 0};

  h = max(f1, f2);
  BOOST_CHECK(table_properties(h, {2, 2}));
  BOOST_CHECK(range_equal(h, fmax));

  h = min(f1, f2);
  BOOST_CHECK(table_properties(h, {2, 2}));
  BOOST_CHECK(range_equal(h, fmin));

  h = weighted_update(f1, f2, 0.3);
  for (std::size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h[i], 0.7 * f1[i] + 0.3 * f2[i], 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_collapse) {
  ltable f({2, 3}, {0, 1, 2, 3, 5, 6});
  ltable h;
  uint_vector vec;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};
  std::vector<double> reordered = {0, 2, 5, 1, 3, 6};

  h = f.maximum(1);
  BOOST_CHECK(table_properties(h, {3}));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.max().lv, 6.0);
  BOOST_CHECK_EQUAL(f.max(vec).lv, 6.0);
  BOOST_CHECK_EQUAL(vec[0], 1);
  BOOST_CHECK_EQUAL(vec[1], 2);

  h = f.minimum(1);
  BOOST_CHECK(table_properties(h, {3}));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.min().lv, 0.0);
  BOOST_CHECK_EQUAL(f.min(vec).lv, 0.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 0);

  double pxy[] = {1.1, 0.5, 0.1, 0.2, 0.4, 0.0};
  double py[] = {1.6, 0.3, 0.4};
  ltable g({2, 3});
  std::transform(pxy, pxy + 6, g.begin(), logarithm<double>());
  h = g.marginal(1);
  BOOST_CHECK(table_properties(h, {3}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_EQUAL(h, g.marginal(uint_vector({1})));
  BOOST_CHECK_CLOSE(double(g.sum()), std::accumulate(pxy, pxy + 6, 0.0), 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(double(h.sum()), 1.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  ltable f({2, 3}, {0, 1, 2, 3, 5, 6});
  ltable h;

  std::vector<double> fall2 = {5, 6};
  h = f.restrict_tail({2});
  BOOST_CHECK(table_properties(h, {2}));
  BOOST_CHECK(range_equal(h, fall2));

  std::vector<double> f1all = {1, 3, 6};
  h = f.restrict_head({1});
  BOOST_CHECK(table_properties(h, {3}));
  BOOST_CHECK(range_equal(h, f1all));

  std::vector<double> fall0 = {0, 1};
  h = f.restrict({1}, {0});
  BOOST_CHECK(table_properties(h, {2}));
  BOOST_CHECK(range_equal(h, fall0));

  std::vector<double> f12 = {6};
  h = f.restrict({1, 0}, {2, 1}); // intentionally reordered
  BOOST_CHECK(table_properties(h, {}));
  BOOST_CHECK(range_equal(h, f12));
}


BOOST_AUTO_TEST_CASE(test_sample) {
  ltable f({2, 3}, {0, 0.1, 0.2, 0.3, 0.5, 0.6});
  f.normalize();
  std::mt19937 rng1;
  std::mt19937 rng2;

  // test marginal sample
  auto fd = f.distribution();
  for (std::size_t i = 0; i < 20; ++i) {
    uint_vector sample = fd(rng1);
    BOOST_CHECK_EQUAL(f.sample(rng2), sample);
  }

  // test conditional sample
  ltable g = f.conditional(1);
  auto gd = g.distribution();
  for (std::size_t xv = 0; xv < 2; ++xv) {
    uint_vector tail = {xv};
    for (std::size_t i = 0; i < 20; ++i) {
      uint_vector sample = gd(rng1, tail);
      BOOST_CHECK_EQUAL(g.restrict_tail(tail).sample(rng2), sample);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  ptable pxy({2, 2}, {0.1, 0.2, 0.3, 0.4});
  ptable qxy({2, 2}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  ltable p = pxy.logarithmic();
  ltable q = qxy.logarithmic();
  ltable m = (p+q) / logd(2);
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
  BOOST_CHECK_CLOSE(p.entropy(uint_vector{0}), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy(1), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information(0, 1), klpq, 1e-6);
  BOOST_CHECK_CLOSE(cross_entropy(p, q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(kl_divergence(p, q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(js_divergence(p, q), jspq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}
