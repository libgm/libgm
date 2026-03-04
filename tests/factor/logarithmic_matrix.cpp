#define BOOST_TEST_MODULE logarithmic_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_matrix.hpp>

#include <libgm/iterator/uint_vector_iterator.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"

namespace libgm {
  template class logarithmic_matrix<double>;
  template class logarithmic_matrix<float>;
}

using namespace libgm;

using lvector = logarithmic_vector<>;
using lmatrix = logarithmic_matrix<>;
using pvector = probability_vector<>;
using pmatrix = probability_matrix<>;
using ltable = logarithmic_table<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  lmatrix a(2, 3);
  BOOST_CHECK(matrix_properties(a, 2, 3));

  lmatrix b(2, 3, logd(3.0));
  BOOST_CHECK(matrix_properties(b, 2, 3));
  BOOST_CHECK_CLOSE(b(0, 0).lv, std::log(3.0), 1e-8);
  BOOST_CHECK_CLOSE(b(1, 2).lv, std::log(3.0), 1e-8);

  lmatrix c(dense_matrix<>::Constant(2, 3, 5.0));
  BOOST_CHECK(matrix_properties(c, 2, 3));
  BOOST_CHECK_EQUAL(std::count(c.begin(), c.end(), 5.0), 6);

  lmatrix d(2, 3, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(matrix_properties(d, 2, 3));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(d[i], i + 1.0, 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  lmatrix f;
  lmatrix g;
  lmatrix h;
  g = pmatrix(2, 3, std::exp(1.0)).logarithmic();
  BOOST_CHECK(matrix_properties(g, 2, 3));
  BOOST_CHECK_CLOSE(g[0], 1.0, 1e-6);
  BOOST_CHECK_CLOSE(g[5], 1.0, 1e-6);

  h.reset(3, 2);
  h.param().fill(2.0);
  BOOST_CHECK(matrix_properties(h, 3, 2));
  BOOST_CHECK_EQUAL(h[0], 2.0);

  swap(g, h);
  BOOST_CHECK(matrix_properties(g, 3, 2));
  BOOST_CHECK(matrix_properties(h, 2, 3));
  BOOST_CHECK_EQUAL(g[0], 2.0);
  BOOST_CHECK_CLOSE(h[0], 1.0, 1e-8);

  f = ltable({3, 2}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(matrix_properties(f, 3, 2));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], (6-i) * 0.1, 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  lmatrix f({2, 3});
  std::iota(f.begin(), f.end(), 1);

  BOOST_CHECK_CLOSE(f(0,0).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1,0).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0,1).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1,1).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0,2).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1,2).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_vector{0,0}).lv, 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}).lv, 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}).lv, 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}).lv, 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}).lv, 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}).lv, 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(0,2), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), 5.0, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_operators) {
  lmatrix f(2, 3, {0, 1, 2, 3, 4, 5});
  lvector g({3, 4});
  lmatrix h;
  h = f.head(1) * g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + g.log(v[0]), 1e-8);
  }

  h.head() *= g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 2*g.log(v[0]), 1e-8);
  }

  h = f.dim(0) / g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) - g.log(v[0]), 1e-8);
  }

  h /= f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), -g.log(v[0]), 1e-8);
  }

  h = f * logd(2.0, log_tag());
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 2.0, 1e-8);
  }

  h *= logd(1.0, log_tag());
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 3.0, 1e-8);
  }

  h = logd(2.0, log_tag()) * f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 2.0, 1e-8);
  }

  h /= logd(1.0, log_tag());
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) + 1.0, 1e-8);
  }

  h = f / logd(2.0, log_tag());
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), f.log(v) - 2.0, 1e-8);
  }

  h = logd(2.0, log_tag()) / f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), 2.0 - f.log(v), 1e-8);
  }

  h = pow(f, 2.0);
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h.log(v), 2.0 * f.log(v), 1e-8);
  }

  lmatrix f1(2, 2, {0, 1, 2, 3});
  lmatrix f2(2, 2, {-2, 3, 0, 0});
  std::vector<double> fmax = {0, 3, 2, 3};
  std::vector<double> fmin = {-2, 1, 0, 0};

  h = max(f1, f2);
  BOOST_CHECK(matrix_properties(h, 2, 2));
  BOOST_CHECK(range_equal(h, fmax));

  h = min(f1, f2);
  BOOST_CHECK(matrix_properties(h, 2, 2));
  BOOST_CHECK(range_equal(h, fmin));

  h = weighted_update(f1, f2, 0.3);
  for (std::size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h[i], 0.7 * f1[i] + 0.3 * f2[i], 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_collapse) {
  lmatrix f(2, 3, {0, 1, 2, 3, 5, 6});
  lvector h, h1;
  std::size_t row, col;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum(1);
  h1 = f.head().max();
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.max().lv, 6.0);
  BOOST_CHECK_EQUAL(f.max(row, col).lv, 6.0);
  BOOST_CHECK_EQUAL(row, 1);
  BOOST_CHECK_EQUAL(col, 2);
  BOOST_CHECK_EQUAL(h, h1);

  h = f.minimum(1);
  h1 = f.head().min();
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.min().lv, 0.0);
  BOOST_CHECK_EQUAL(f.min(row, col).lv, 0.0);
  BOOST_CHECK_EQUAL(row, 0);
  BOOST_CHECK_EQUAL(col, 0);

  pmatrix pxy(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  pvector py({1.6, 0.3, 0.4});
  lmatrix g = pxy.logarithmic();
  h = g.marginal(1);
  h1 = g.head().sum();
  BOOST_CHECK(vector_properties(h, 3));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(std::exp(h[i]), py[i], 1e-7);
  }
  BOOST_CHECK_EQUAL(h, h1);
  BOOST_CHECK_CLOSE(double(g.sum()), pxy.sum(), 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(double(h.sum()), 1.0, 1e-8);

  /*
  h = f.exp_log(py);
  BOOST_CHECK(vector_properties(h, {x}));
  BOOST_CHECK_CLOSE(h[0], 0*1.6 + 2*0.3 + 5*0.4, 1e-8);
  BOOST_CHECK_CLOSE(h[1], 1*1.6 + 3*0.3 + 6*0.4, 1e-8);
  */
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  lmatrix f(2, 3, {0, 1, 2, 3, 5, 6});
  lvector h;

  std::vector<double> ft = {5, 6};
  h = f.restrict_tail(2);
  BOOST_CHECK(vector_properties(h, 2));
  BOOST_CHECK(range_equal(h, ft));

  std::vector<double> fh = {0, 2, 5};
  h = f.restrict_head(0);
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, fh));

  std::vector<double> fr = {1, 3, 6};
  h = f.restrict(0, 1);
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, fr));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;

  pmatrix pxy(2, 2, {0.1, 0.2, 0.3, 0.4});
  pmatrix qxy(2, 2, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
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
  BOOST_CHECK_CLOSE(p.entropy(0), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.entropy(1), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information(0, 1), klpq, 1e-6);
  BOOST_CHECK_CLOSE(p.mutual_information(), klpq, 1e-6);
  BOOST_CHECK_CLOSE(cross_entropy(p, q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(kl_divergence(p, q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(js_divergence(p, q), jspq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}
