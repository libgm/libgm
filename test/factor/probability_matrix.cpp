#define BOOST_TEST_MODULE probability_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_matrix.hpp>

#include <libgm/iterator/uint_vector_iterator.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"

namespace libgm {
  template class probability_matrix<double>;
  template class probability_matrix<float>;
}

using namespace libgm;

using lmatrix = logarithmic_matrix<>;
using pvector = probability_vector<>;
using pmatrix = probability_matrix<>;
using ptable = probability_table<>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  pmatrix a(2, 3);
  BOOST_CHECK(matrix_properties(a, 2, 3));

  pmatrix b(2, 3, 3.0);
  BOOST_CHECK(matrix_properties(b, 2, 3));
  BOOST_CHECK_CLOSE(b(0, 0), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(b(1, 2), 3.0, 1e-8);

  pmatrix c(dense_matrix<>::Constant(2, 3, 5.0));
  BOOST_CHECK(matrix_properties(c, 2, 3));
  BOOST_CHECK_EQUAL(std::count(c.begin(), c.end(), 5.0), 6);

  pmatrix d(2, 3, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(matrix_properties(d, 2, 3));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(d[i], i + 1.0, 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  pmatrix g;
  pmatrix h;
  g = pmatrix(2, 3, 1.0);
  BOOST_CHECK(matrix_properties(g, 2, 3));
  BOOST_CHECK_EQUAL(g[0], 1.0);

  h.reset(3, 2);
  h.param().fill(2.0);
  BOOST_CHECK(matrix_properties(h, 3, 2));
  BOOST_CHECK_EQUAL(h[0], 2.0);

  swap(g, h);
  BOOST_CHECK(matrix_properties(g, 3, 2));
  BOOST_CHECK(matrix_properties(h, 2, 3));
  BOOST_CHECK_EQUAL(g[0], 2.0);
  BOOST_CHECK_EQUAL(h[0], 1.0);

  pmatrix f = lmatrix(2, 3, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).probability();
  BOOST_CHECK(matrix_properties(f, 2, 3));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], std::exp((i+1) * 0.1), 1e-8);
  }

  f = ptable({3, 2}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(matrix_properties(f, 3, 2));
  for (std::size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(f[i], (6-i) * 0.1, 1e-8);
  }
}


BOOST_AUTO_TEST_CASE(test_indexing) {
  pmatrix f(2, 3);
  std::iota(f.begin(), f.end(), 1);

  BOOST_CHECK_CLOSE(f(0, 0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 0), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0, 1), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 1), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(0, 2), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(1, 2), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f(uint_vector{0,0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{0,2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f(uint_vector{1,2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log(0, 2), std::log(5.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(uint_vector{0,2}), std::log(5.0), 1e-8);
}


BOOST_AUTO_TEST_CASE(test_operators) {
  pmatrix f(2, 3, {0, 1, 2, 3, 4, 5});
  pvector g({3, 4});
  pmatrix h;
  h = f.head(1) * g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * g(v[0]), 1e-8);
  }

  h.head() *= g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * g(v[0]) * g(v[0]), 1e-8);
  }

  h = f.dim(0) / g;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) / g(v[0]), 1e-8);
  }

  h /= f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    if (f(v)) {
      BOOST_CHECK_CLOSE(h(v), 1.0 / g(v[0]), 1e-8);
    } else {
      BOOST_CHECK(std::isnan(h(v)));
    }
  }

  h = f * 2.0;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 2.0, 1e-8);
  }

  h *= 3.0;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 6.0, 1e-8);
  }

  h = 2.0 * f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 2.0, 1e-8);
  }

  h /= 4.0;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) * 0.5, 1e-8);
  }

  h = f / 3.0;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), f(v) / 3.0, 1e-8);
  }

  h = 3.0 / f;
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    if (f(v)) {
      BOOST_CHECK_CLOSE(h(v), 3.0 / f(v), 1e-8);
    } else {
      BOOST_CHECK(std::isinf(h(v)));
    }
  }

  h = pow(f, 3.0);
  BOOST_CHECK(matrix_properties(h, 2, 3));
  for (const uint_vector& v : uint_vectors({2, 3})) {
    BOOST_CHECK_CLOSE(h(v), std::pow(f(v), 3.0), 1e-8);
  }

  pmatrix f1(2, 2, {0, 1, 2, 3});
  pmatrix f2(2, 2, {-2, 3, 0, 0});
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
  pmatrix f(2, 3, {0, 1, 2, 3, 5, 6});
  pvector h, h1;
  std::size_t row, col;

  std::vector<double> hmax = {1, 3, 6};
  std::vector<double> hmin = {0, 2, 5};

  h = f.maximum(1);
  h1 = f.head().max();
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, hmax));
  BOOST_CHECK_EQUAL(f.max(), 6.0);
  BOOST_CHECK_EQUAL(f.max(row, col), 6.0);
  BOOST_CHECK_EQUAL(row, 1);
  BOOST_CHECK_EQUAL(col, 2);
  BOOST_CHECK_EQUAL(h, h1);

  h = f.minimum(1);
  h1 = f.head().min();
  BOOST_CHECK(vector_properties(h, 3));
  BOOST_CHECK(range_equal(h, hmin));
  BOOST_CHECK_EQUAL(f.min(), 0.0);
  BOOST_CHECK_EQUAL(f.min(row, col), 0.0);
  BOOST_CHECK_EQUAL(row, 0);
  BOOST_CHECK_EQUAL(col, 0);

  pmatrix pxy(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  pvector py({1.6, 0.3, 0.4});
  h = pxy.marginal(1);
  h1 = pxy.head().sum();
  BOOST_CHECK(vector_properties(h, 3));
  for (std::size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(h[i], py[i], 1e-7);
  }
  BOOST_CHECK_EQUAL(h, h1);
  BOOST_CHECK_CLOSE(pxy.sum(), 1.1+0.5+0.1+0.2+0.4, 1e-8);
  h.normalize();
  BOOST_CHECK_CLOSE(h.sum(), 1.0, 1e-8);

  pvector qx({0.4, 0.6});
  pvector qy({0.2, 0.5, 0.3});

  pvector rx = (pxy.dim(1) * qy).marginal(0);
  pvector rx1 = (pxy.tail() * qy).tail().sum();
  BOOST_CHECK(vector_properties(rx, 2));
  BOOST_CHECK_CLOSE(rx[0], 1.1*0.2+0.1*0.5+0.4*0.3, 1e-8);
  BOOST_CHECK_CLOSE(rx[1], 0.5*0.2+0.2*0.5+0.0*0.3, 1e-8);
  BOOST_CHECK_EQUAL(rx, rx1);

  pvector ry = (qx * pxy.dim(0)).marginal(1);
  pvector ry1 = (qx * pxy.head()).head().sum();
  BOOST_CHECK(vector_properties(ry, 3));
  BOOST_CHECK_CLOSE(ry[0], 1.1*0.4 + 0.5*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[1], 0.1*0.4 + 0.2*0.6, 1e-8);
  BOOST_CHECK_CLOSE(ry[2], 0.4*0.4 + 0.0*0.6, 1e-8);
  BOOST_CHECK_EQUAL(ry, ry1);

  /* unsupported
  pvector sx = (pxy.dim(0) * qx).marginal(0);
  pvector sx1 = (pxy.head() * qx).marginal(0);
  BOOST_CHECK(vector_properties(sx, 2));
  BOOST_CHECK_CLOSE(sx[0], (1.1+0.1+0.4)*0.4, 1e-8);
  BOOST_CHECK_CLOSE(sx[1], (0.5+0.2+0.0)*0.6, 1e-8);
  BOOST_CHECK_EQUAL(sx, sx1);
  */
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  pmatrix f(2, 3, {0, 1, 2, 3, 5, 6});
  pvector h;

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

  pmatrix p(2, 2, {0.1, 0.2, 0.3, 0.4});
  pmatrix q(2, 2, {0.4*0.3, 0.6*0.3, 0.4*0.7, 0.6*0.7});
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
