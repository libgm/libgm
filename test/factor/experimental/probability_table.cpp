#define BOOST_TEST_MODULE probability_table
#include <boost/test/unit_test.hpp>

#include <libgm/factor/experimental/probability_table.hpp>

#include <libgm/datastructure/uint_vector_iterator.hpp>
#include <libgm/factor/experimental/logarithmic_table.hpp>
//#include <libgm/factor/experimental/probability_vector.hpp>
//#include <libgm/factor/experimental/probability_matrix.hpp>

#include "predicates.hpp"

namespace libgm { namespace experimental {
  template class probability_table<double>;
  template class probability_table<float>;
} }

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector)

typedef experimental::logarithmic_table<> ltable;
typedef experimental::probability_table<> ptable;

BOOST_AUTO_TEST_CASE(test_constructors) {
  ptable a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.arity() == 0);

  ptable b({2, 3});
  BOOST_CHECK(table_properties(b, {2, 3}));

  ptable c(2.0);
  BOOST_CHECK(table_properties(c, {}));
  BOOST_CHECK_CLOSE(c[0], 2.0, 1e-8);

  ptable d({2}, 3.0);
  BOOST_CHECK(table_properties(d, {2}));
  BOOST_CHECK_CLOSE(d[0], 3.0, 1e-8);
  BOOST_CHECK_CLOSE(d[1], 3.0, 1e-8);

  table<double> params({2, 3}, 5.0);
  ptable f(params);
  BOOST_CHECK(table_properties(f, {2, 3}));
  BOOST_CHECK_EQUAL(std::count(f.begin(), f.end(), 5.0), 6);

  ptable g({2}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, {2}));
  BOOST_CHECK_EQUAL(g[0], 6.0);
  BOOST_CHECK_EQUAL(g[1], 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  ptable f;
  f = 2.0;
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], 2.0, 1e-8);

  f.reset({2, 3});
  BOOST_CHECK(table_properties(f, {2, 3}));

  f = 3.0;
  BOOST_CHECK(table_properties(f, {}));
  BOOST_CHECK_CLOSE(f[0], 3.0, 1e-8);

  ltable ct({2}, logd(0.5));
  f = ct.probability();
  BOOST_CHECK(table_properties(f, {2}));
  BOOST_CHECK_CLOSE(f[0], 0.5, 1e-8);
  BOOST_CHECK_CLOSE(f[1], 0.5, 1e-8);

  ptable g({2, 3});
  swap(f, g);
  BOOST_CHECK(table_properties(f, {2, 3}));
  BOOST_CHECK(table_properties(g, {2}));
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  ptable f({2, 3});
  std::iota(f.begin(), f.end(), 1);
  BOOST_CHECK_CLOSE(f({0,0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0,1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1,2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log({0,2}), std::log(5.0), 1e-8);
}


BOOST_AUTO_TEST_CASE(test_operators) {
  ptable f({2, 2}, {0, 1, 2, 3});       // x, y
  ptable g({2, 3}, {1, 2, 3, 4, 5, 6}); // y, z
  ptable h;
  h = f.wise(1) * g.wise(0);
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f({v[0],v[1]}) * g({v[1],v[2]}), 1e-8);
  }

  h.wise({1, 2}) *= g;
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f({v[0],v[1]}) * std::pow(g({v[1],v[2]}), 2), 1e-8);
  }

  h = f.wise(1) / g.wise(0);
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f({v[0],v[1]}) / g({v[1],v[2]}), 1e-8);
  }

  h.wise({0, 1}) /= f;
  BOOST_CHECK(table_properties(h, {2, 2, 3}));
  for (const uint_vector& v : uint_vectors({2, 2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f({v[0],v[1]}) ? (1 / g({v[1],v[2]})) : 0.0, 1e-8);
  }

  h = f * 2.0;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 2.0, 1e-8);
  }

  h *= 3.0;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 6.0, 1e-8);
  }

  h = 2.0 * f;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 2.0, 1e-8);
  }


  h /= 4.0;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 0.5, 1e-8);
  }

  h = f / 3.0;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), f(v) / 3.0, 1e-8);
  }

  h = 3.0 / f;
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    if (f(v)) {
      BOOST_CHECK_CLOSE(h(v), 3.0 / f(v), 1e-8);
    } else {
      BOOST_CHECK(std::isinf(h(v)));
    }
  }

  h = pow(f, 3.0);
  BOOST_CHECK(table_properties(h, {2, 2}));
  for (const uint_vector& v : uint_vectors({2, 2})) {
    BOOST_CHECK_CLOSE(h(v), std::pow(f(v), 3.0), 1e-8);
  }

  ptable f1({2, 2}, {0, 1, 2, 3});
  ptable f2({2, 2}, {-2, 3, 0, 0});
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
  ptable f({2, 3}, {0, 1, 2, 3, 5, 6});
  ptable h;
  uint_vector vec;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};
  std::vector<double> reordered = {0, 2, 5, 1, 3, 6};

  h = f.maximum(1);
  BOOST_CHECK(table_properties(h, {3}));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.maximum(), 6.0);
  BOOST_CHECK_EQUAL(f.maximum(&vec), 6.0);
  BOOST_CHECK_EQUAL(vec[0], 1);
  BOOST_CHECK_EQUAL(vec[1], 2);

  h = f.minimum(1);
  BOOST_CHECK(table_properties(h, {3}));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.minimum(), 0.0);
  BOOST_CHECK_EQUAL(f.minimum(&vec), 0.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 0);

  h = f.maximum({1, 0});
  BOOST_CHECK(table_properties(h, {3, 2}));
  BOOST_CHECK(range_equal(h, reordered));

  h = ptable();
  h = f.minimum({1, 0});
  BOOST_CHECK(table_properties(h, {3, 2}));
  BOOST_CHECK(range_equal(h, reordered));

  ptable fx({2}, {0.5, 0.2});
  ptable fxy({2, 3}, {2.2, 2.5, 0.2, 1.0, 0.8, 0.0});
  ptable pxy({2, 3}, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  ptable py({3}, {1.6, 0.3, 0.4});
  ptable g = (fx * fxy.wise(0)).marginal(1);
  h = pxy.marginal(1);
  BOOST_CHECK(table_properties(g, {3}));
  BOOST_CHECK(table_properties(h, {3}));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(g[i], py[i], 1e-7);
    BOOST_CHECK_CLOSE(h[i], py[i], 1e-7);
  }
  BOOST_CHECK_CLOSE(pxy.marginal(), 1.1+0.5+0.1+0.2+0.4, 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(h.marginal(), 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  ptable f({2, 3}, {0, 1, 2, 3, 5, 6});
  ptable h;

  std::vector<double> fall2 = {5, 6};
  h = f.restrict_head({2});
  BOOST_CHECK(table_properties(h, {2}));
  BOOST_CHECK(range_equal(h, fall2));

  std::vector<double> f1all = {1, 3, 6};
  h = f.restrict_tail({1});
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
  ptable f({2, 3}, {0, 1, 2, 3, 5, 6});
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
  ptable g = f.wise(0) / f.marginal(0); // f.conditional(0);
  auto gd = g.distribution(1);
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
  ptable p({2, 2}, {0.1, 0.2, 0.3, 0.4});
  ptable q({2, 2}, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
  ptable m = (p+q) / 2.0;
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
  BOOST_CHECK_CLOSE(p.entropy(0), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy({1}), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information(0, 1), klpq, 1e-6);
  BOOST_CHECK_CLOSE(cross_entropy(p, q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(kl_divergence(p, q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(js_divergence(p, q), jspq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}

