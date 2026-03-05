#define BOOST_TEST_MODULE logarithmic_matrix
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_matrix.hpp>

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LVector = LogarithmicVector<double>;
using LMatrix = LogarithmicMatrix<double>;
using PVector = ProbabilityVector<double>;
using PMatrix = ProbabilityMatrix<double>;
using LTable = LogarithmicTable<double>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  LMatrix a(2, 3);
  BOOST_CHECK(matrix_properties(a, 2, 3));

  LMatrix b(2, 3, Exp<double>(3.0));
  BOOST_CHECK(matrix_properties(b, 2, 3));
  BOOST_CHECK_EQUAL(b.log(0, 0), 3.0);
  BOOST_CHECK_EQUAL(b.log(1, 2), 3.0);

  LMatrix c(Eigen::Matrix<double, 2, 3>::Constant(5.0));
  BOOST_CHECK(matrix_properties(c, 2, 3));
  BOOST_CHECK_EQUAL(c.log(0, 0), 5.0);
  BOOST_CHECK_EQUAL(c.log(1, 2), 5.0);

  LMatrix d(2, 3, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(matrix_properties(d, 2, 3));
  BOOST_CHECK_EQUAL(d.log(0, 0), 1.0);
  BOOST_CHECK_EQUAL(d.log(1, 2), 6.0);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap_and_conversion) {
  LMatrix g = PMatrix(2, 3, std::exp(1.0)).logarithmic();
  BOOST_CHECK(matrix_properties(g, 2, 3));
  BOOST_CHECK_CLOSE(g.log(0, 0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(g.log(1, 2), 1.0, 1e-8);

  LMatrix h(3, 2, Exp<double>(2.0));
  swap(g, h);
  BOOST_CHECK(matrix_properties(g, 3, 2));
  BOOST_CHECK(matrix_properties(h, 2, 3));
  BOOST_CHECK_EQUAL(g.log(0, 0), 2.0);
  BOOST_CHECK_CLOSE(h.log(0, 0), 1.0, 1e-8);

  LMatrix f = LTable({3, 2}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}).matrix();
  BOOST_CHECK(matrix_properties(f, 3, 2));
  BOOST_CHECK_EQUAL(f.log(0, 0), 0.6);
  BOOST_CHECK_EQUAL(f.log(2, 1), 0.1);

  LTable t = f.table();
  BOOST_CHECK(table_properties(t, Shape{3, 2}));
  BOOST_CHECK_EQUAL(t.log({0, 0}), 0.6);
  BOOST_CHECK_EQUAL(t.log({2, 1}), 0.1);
}

BOOST_AUTO_TEST_CASE(test_indexing_and_ops) {
  LMatrix f(2, 3, {1, 2, 3, 4, 5, 6});
  LMatrix g(2, 3, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});

  BOOST_CHECK_EQUAL(f.log(std::vector<size_t>{1, 2}), 6.0);
  BOOST_CHECK_EQUAL(log(f(0, 2)), 5.0);

  BOOST_CHECK_SMALL(max_diff(f * Exp<double>(2.0), LMatrix(2, 3, {3, 4, 5, 6, 7, 8})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(Exp<double>(2.0) * f, LMatrix(2, 3, {3, 4, 5, 6, 7, 8})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / Exp<double>(2.0), LMatrix(2, 3, {-1, 0, 1, 2, 3, 4})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(Exp<double>(6.0) / f, LMatrix(2, 3, {5, 4, 3, 2, 1, 0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f * g, LMatrix(2, 3, {1.5, 3, 4.5, 6, 7.5, 9})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, LMatrix(2, 3, {0.5, 1, 1.5, 2, 2.5, 3})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), LMatrix(2, 3, {2, 4, 6, 8, 10, 12})), 1e-8);

  LMatrix h = f;
  h *= Exp<double>(2.0);
  h /= Exp<double>(2.0);
  BOOST_CHECK_SMALL(max_diff(h, f), 1e-8);
  h *= g;
  h /= g;
  BOOST_CHECK_SMALL(max_diff(h, f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_join) {
  LMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});
  LVector qx({0.4, 0.6});
  LVector qy({0.2, 0.5, 0.3});

  LMatrix fx = f.multiply_front(qx);
  LMatrix fy = f.multiply_back(qy);
  BOOST_CHECK_CLOSE(fx.log(1, 0), f.log(1, 0) + qx.log(1), 1e-8);
  BOOST_CHECK_CLOSE(fy.log(0, 2), f.log(0, 2) + qy.log(2), 1e-8);
  LMatrix fix = f;
  fix.multiply_in_front(qx);
  BOOST_CHECK_SMALL(max_diff(fix, fx), 1e-8);
  LMatrix fiy = f;
  fiy.multiply_in_back(qy);
  BOOST_CHECK_SMALL(max_diff(fiy, fy), 1e-8);

  LMatrix dx = f.divide_front(qx);
  LMatrix dy = f.divide_back(qy);
  BOOST_CHECK_CLOSE(dx.log(1, 1), f.log(1, 1) - qx.log(1), 1e-8);
  BOOST_CHECK_CLOSE(dy.log(0, 1), f.log(0, 1) - qy.log(1), 1e-8);
  LMatrix dix = f;
  dix.divide_in_front(qx);
  BOOST_CHECK_SMALL(max_diff(dix, dx), 1e-8);
  LMatrix diy = f;
  diy.divide_in_back(qy);
  BOOST_CHECK_SMALL(max_diff(diy, dy), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_collapse) {
  LMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});

  LVector mx = f.maximum_front();
  LVector my = f.maximum_back();
  BOOST_CHECK_EQUAL(mx.log(0), 1.1);
  BOOST_CHECK_EQUAL(mx.log(1), 0.5);
  BOOST_CHECK_EQUAL(my.log(0), 1.1);
  BOOST_CHECK_EQUAL(my.log(2), 0.4);

  LVector mnx = f.minimum_front();
  LVector mny = f.minimum_back();
  BOOST_CHECK_EQUAL(mnx.log(0), 0.1);
  BOOST_CHECK_EQUAL(mnx.log(1), 0.0);
  BOOST_CHECK_EQUAL(mny.log(0), 0.5);
  BOOST_CHECK_EQUAL(mny.log(2), 0.0);

  std::vector<size_t> argmax;
  std::vector<size_t> argmin;
  BOOST_CHECK_EQUAL(log(f.maximum(&argmax)), 1.1);
  BOOST_CHECK_EQUAL(log(f.minimum(&argmin)), 0.0);
  BOOST_CHECK(argmax == std::vector<size_t>({0, 0}));
  BOOST_CHECK(argmin == std::vector<size_t>({1, 2}));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  LMatrix f(2, 3, {1.1, 0.5, 0.1, 0.2, 0.4, 0.0});

  LVector rfront = f.restrict_front({1});
  LVector rback = f.restrict_back({2});
  BOOST_CHECK(vector_properties(rfront, 3));
  BOOST_CHECK(vector_properties(rback, 2));
  BOOST_CHECK_EQUAL(rfront.log(0), 0.5);
  BOOST_CHECK_EQUAL(rfront.log(2), 0.0);
  BOOST_CHECK_EQUAL(rback.log(0), 0.4);
  BOOST_CHECK_EQUAL(rback.log(1), 0.0);

  LMatrix tr = f.transpose();
  BOOST_CHECK(matrix_properties(tr, 3, 2));
  BOOST_CHECK_EQUAL(tr.log(0, 1), f.log(1, 0));
  BOOST_CHECK_EQUAL(tr.log(2, 0), f.log(0, 2));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;

  PMatrix pxy(2, 2, {0.1, 0.2, 0.3, 0.4});
  PMatrix qxy(2, 2, {0.4 * 0.3, 0.6 * 0.3, 0.4 * 0.7, 0.6 * 0.7});
  LMatrix p = pxy.logarithmic();
  LMatrix q = qxy.logarithmic();
  double hp = -(0.1 * log(0.1) + 0.2 * log(0.2) + 0.3 * log(0.3) + 0.4 * log(0.4));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      const double pv = pxy(i, j);
      const double qv = qxy(i, j);
      hpq += -pv * log(qv);
      klpq += pv * log(pv / qv);
      const double d = std::abs(log(pv) - log(qv));
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
  LMatrix f(2, 2, {1.0, 2.0, 3.0, 4.0});

  // mutable + const param access
  f.param()(0, 1) = 7.0;
  const LMatrix& cf = f;
  BOOST_CHECK_EQUAL(cf.param()(0, 1), 7.0);

  // copy/move ctor + assignment
  LMatrix copy_ctor(f);
  BOOST_CHECK_SMALL(max_diff(copy_ctor, f), 1e-8);

  LMatrix copy_assign;
  copy_assign = f;
  BOOST_CHECK_SMALL(max_diff(copy_assign, f), 1e-8);

  LMatrix move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(max_diff(move_ctor, f), 1e-8);

  LMatrix move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(max_diff(move_assign, f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_edge_zero_behavior) {
  PMatrix pz(2, 2, {0.0, 1.0, 2.0, 0.0});
  LMatrix z = pz.logarithmic();
  BOOST_CHECK(std::isinf(z.log(0, 0)));
  BOOST_CHECK(std::isinf(z.log(1, 1)));

  LMatrix inv = Exp<double>(3.0) / z;
  BOOST_CHECK(std::isinf(inv.log(0, 0)));
  BOOST_CHECK_CLOSE(inv.log(0, 1), 2.3068528194400546, 1e-8); // 3 - log(2)
  BOOST_CHECK_CLOSE(inv.log(1, 0), 3.0, 1e-8);
  BOOST_CHECK(std::isinf(inv.log(1, 1)));
}
