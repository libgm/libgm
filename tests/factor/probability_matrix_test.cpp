#define BOOST_TEST_MODULE probability_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_matrix.hpp>

#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LMatrix = LogarithmicMatrix<double>;
using PVector = ProbabilityVector<double>;
using PMatrix = ProbabilityMatrix<double>;
using PTable = ProbabilityTable<double>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  PMatrix a(2, 3);
  BOOST_CHECK(matrix_properties(a, 2, 3));

  PMatrix b(2, 3, 3.0);
  BOOST_CHECK(matrix_properties(b, 2, 3));
  BOOST_CHECK_EQUAL(b(0, 0), 3.0);
  BOOST_CHECK_EQUAL(b(1, 2), 3.0);

  PMatrix c(Eigen::Matrix<double, 2, 3>::Constant(5.0));
  BOOST_CHECK(matrix_properties(c, 2, 3));
  BOOST_CHECK_EQUAL(c(0, 0), 5.0);
  BOOST_CHECK_EQUAL(c(1, 2), 5.0);

  PMatrix d(2, 3, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(matrix_properties(d, 2, 3));
  BOOST_CHECK_EQUAL(d(0, 0), 1.0);
  BOOST_CHECK_EQUAL(d(1, 2), 6.0);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap_and_conversion) {
  PMatrix g(2, 3, 1.0);
  PMatrix h(3, 2, 2.0);

  swap(g, h);
  BOOST_CHECK(matrix_properties(g, 3, 2));
  BOOST_CHECK(matrix_properties(h, 2, 3));
  BOOST_CHECK_EQUAL(g(0, 0), 2.0);
  BOOST_CHECK_EQUAL(h(0, 0), 1.0);

  PMatrix f = LMatrix(2, 3, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).probability();
  BOOST_CHECK(matrix_properties(f, 2, 3));
  BOOST_CHECK_CLOSE(f(0, 0), std::exp(0.1), 1e-8);
  BOOST_CHECK_CLOSE(f(1, 2), std::exp(0.6), 1e-8);

  f = PTable({3, 2}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(matrix_properties(f, 3, 2));
  BOOST_CHECK_EQUAL(f(0, 0), 0.6);
  BOOST_CHECK_EQUAL(f(2, 1), 0.1);

  PTable t = f.table();
  BOOST_CHECK(table_properties(t, Shape{3, 2}));
  BOOST_CHECK_EQUAL(t({0, 0}), 0.6);
  BOOST_CHECK_EQUAL(t({2, 1}), 0.1);
}

BOOST_AUTO_TEST_CASE(test_indexing_and_ops) {
  PMatrix f(2, 3, {1, 2, 3, 4, 5, 6});
  PMatrix g(2, 3, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});

  BOOST_CHECK_EQUAL(f(std::vector<size_t>{1, 2}), 6.0);
  BOOST_CHECK_CLOSE(f.log(1, 2), std::log(6.0), 1e-8);
  BOOST_CHECK_CLOSE(f.log(std::vector<size_t>{0, 2}), std::log(5.0), 1e-8);

  BOOST_CHECK_SMALL(max_diff(f * 2.0, PMatrix(2, 3, {2, 4, 6, 8, 10, 12})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(2.0 * f, PMatrix(2, 3, {2, 4, 6, 8, 10, 12})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / 2.0, PMatrix(2, 3, {0.5, 1, 1.5, 2, 2.5, 3})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(6.0 / f, PMatrix(2, 3, {6, 3, 2, 1.5, 1.2, 1})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f * g, PMatrix(2, 3, {0.5, 2, 4.5, 8, 12.5, 18})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, PMatrix(2, 3, {2, 2, 2, 2, 2, 2})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), PMatrix(2, 3, {1, 4, 9, 16, 25, 36})), 1e-8);

  PMatrix h = f;
  h *= 2.0;
  h /= 2.0;
  BOOST_CHECK_SMALL(max_diff(h, f), 1e-8);
  h *= g;
  h /= g;
  BOOST_CHECK_SMALL(max_diff(h, f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_join) {
  PMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  PVector qx({0.4, 0.6});
  PVector qy({0.2, 0.5, 0.3});

  PMatrix fx = f.multiply_front(qx);
  PMatrix fy = f.multiply_back(qy);
  BOOST_CHECK_CLOSE(fx(1, 0), f(1, 0) * qx(1), 1e-8);
  BOOST_CHECK_CLOSE(fy(0, 2), f(0, 2) * qy(2), 1e-8);
  PMatrix fix = f;
  fix.multiply_in_front(qx);
  BOOST_CHECK_SMALL(max_diff(fix, fx), 1e-8);
  PMatrix fiy = f;
  fiy.multiply_in_back(qy);
  BOOST_CHECK_SMALL(max_diff(fiy, fy), 1e-8);

  PMatrix dx = f.divide_front(qx);
  PMatrix dy = f.divide_back(qy);
  BOOST_CHECK_CLOSE(dx(1, 1), f(1, 1) / qx(1), 1e-8);
  BOOST_CHECK_CLOSE(dy(0, 1), f(0, 1) / qy(1), 1e-8);
  PMatrix dix = f;
  dix.divide_in_front(qx);
  BOOST_CHECK_SMALL(max_diff(dix, dx), 1e-8);
  PMatrix diy = f;
  diy.divide_in_back(qy);
  BOOST_CHECK_SMALL(max_diff(diy, dy), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_collapse) {
  PMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});

  PVector back = f.marginal_back();
  PVector front = f.marginal_front();
  BOOST_CHECK(vector_properties(back, 3));
  BOOST_CHECK(vector_properties(front, 2));
  BOOST_CHECK_CLOSE(back(0), 1.6, 1e-8);
  BOOST_CHECK_CLOSE(back(1), 0.3, 1e-8);
  BOOST_CHECK_CLOSE(back(2), 0.4, 1e-8);
  BOOST_CHECK_CLOSE(front(0), 1.6, 1e-8);
  BOOST_CHECK_CLOSE(front(1), 0.7, 1e-8);

  PVector mx = f.maximum_front();
  PVector my = f.maximum_back();
  BOOST_CHECK_EQUAL(mx(0), 1.1);
  BOOST_CHECK_EQUAL(mx(1), 0.5);
  BOOST_CHECK_EQUAL(my(0), 1.1);
  BOOST_CHECK_EQUAL(my(2), 0.4);

  PVector mnx = f.minimum_front();
  PVector mny = f.minimum_back();
  BOOST_CHECK_EQUAL(mnx(0), 0.1);
  BOOST_CHECK_EQUAL(mnx(1), 0.0);
  BOOST_CHECK_EQUAL(mny(0), 0.5);
  BOOST_CHECK_EQUAL(mny(2), 0.0);

  std::vector<size_t> argmax;
  std::vector<size_t> argmin;
  BOOST_CHECK_EQUAL(f.maximum(&argmax), 1.1);
  BOOST_CHECK_EQUAL(f.minimum(&argmin), 0.0);
  BOOST_CHECK(argmax == std::vector<size_t>({0, 0}));
  BOOST_CHECK(argmin == std::vector<size_t>({1, 2}));

  PMatrix n = f;
  n.normalize();
  BOOST_CHECK_CLOSE(n.marginal(), 1.0, 1e-8);

  PMatrix nh = f;
  nh.normalize_head(1);
  BOOST_CHECK_CLOSE(nh.param().col(0).sum(), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(nh.param().col(1).sum(), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(nh.param().col(2).sum(), 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  PMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});

  PVector rfront = f.restrict_front({1});
  PVector rback = f.restrict_back({2});
  BOOST_CHECK(vector_properties(rfront, 3));
  BOOST_CHECK(vector_properties(rback, 2));
  BOOST_CHECK_EQUAL(rfront(0), 0.5);
  BOOST_CHECK_EQUAL(rfront(2), 0.0);
  BOOST_CHECK_EQUAL(rback(0), 0.4);
  BOOST_CHECK_EQUAL(rback(1), 0.0);

  PMatrix tr = f.transpose();
  BOOST_CHECK(matrix_properties(tr, 3, 2));
  BOOST_CHECK_EQUAL(tr(0, 1), f(1, 0));
  BOOST_CHECK_EQUAL(tr(2, 0), f(0, 2));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;

  PMatrix p(2, 2, {0.1, 0.2, 0.3, 0.4});
  PMatrix q(2, 2, {0.4 * 0.3, 0.6 * 0.3, 0.4 * 0.7, 0.6 * 0.7});
  double hp = -(0.1 * log(0.1) + 0.2 * log(0.2) + 0.3 * log(0.3) + 0.4 * log(0.4));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      const double pv = p(i, j);
      const double qv = q(i, j);
      hpq += -pv * log(qv);
      klpq += pv * log(pv / qv);
      const double d = std::abs(pv - qv);
      sumdiff += d;
      maxdiff = std::max(maxdiff, d);
    }
  }

  BOOST_CHECK_CLOSE(p.entropy(), hp, 1e-8);
  BOOST_CHECK_CLOSE(p.cross_entropy(q), hpq, 1e-8);
  BOOST_CHECK_CLOSE(p.kl_divergence(q), klpq, 1e-8);
  BOOST_CHECK_CLOSE(p.sum_diff(q), sumdiff, 1e-8);
  BOOST_CHECK_CLOSE(p.max_diff(q), maxdiff, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_copy_move_and_param) {
  PMatrix f(2, 2, {1.0, 2.0, 3.0, 4.0});

  // mutable + const param access
  f.param()(0, 1) = 7.0;
  const PMatrix& cf = f;
  BOOST_CHECK_EQUAL(cf.param()(0, 1), 7.0);

  // copy/move ctor + assignment
  PMatrix copy_ctor(f);
  BOOST_CHECK_SMALL(max_diff(copy_ctor, f), 1e-8);

  PMatrix copy_assign;
  copy_assign = f;
  BOOST_CHECK_SMALL(max_diff(copy_assign, f), 1e-8);

  PMatrix move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(max_diff(move_ctor, f), 1e-8);

  PMatrix move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(max_diff(move_assign, f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_edge_zero_behavior) {
  PMatrix z(2, 2, {0.0, 1.0, 2.0, 0.0});
  PMatrix inv = 3.0 / z;
  BOOST_CHECK(std::isinf(inv(0, 0)));
  BOOST_CHECK_CLOSE(inv(0, 1), 1.5, 1e-8);
  BOOST_CHECK_CLOSE(inv(1, 0), 3.0, 1e-8);
  BOOST_CHECK(std::isinf(inv(1, 1)));

  LMatrix lz = z.logarithmic();
  BOOST_CHECK(std::isinf(lz.log(0, 0)));
  BOOST_CHECK(std::isinf(lz.log(1, 1)));
}
