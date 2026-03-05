#define BOOST_TEST_MODULE probability_vector
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_vector.hpp>

#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LVector = LogarithmicVector<double>;
using PVector = ProbabilityVector<double>;
using PTable = ProbabilityTable<double>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  PVector a(3);
  BOOST_CHECK(vector_properties(a, 3));

  PVector b(2, 3.0);
  BOOST_CHECK(vector_properties(b, 2));
  BOOST_CHECK_EQUAL(b(0), 3.0);
  BOOST_CHECK_EQUAL(b(1), 3.0);

  PVector c(Eigen::Vector3d(2.0, 3.0, 4.0));
  BOOST_CHECK(vector_properties(c, 3));
  BOOST_CHECK_EQUAL(c(0), 2.0);
  BOOST_CHECK_EQUAL(c(1), 3.0);
  BOOST_CHECK_EQUAL(c(2), 4.0);

  PVector d = {6.0, 6.5};
  BOOST_CHECK(vector_properties(d, 2));
  BOOST_CHECK_EQUAL(d(0), 6.0);
  BOOST_CHECK_EQUAL(d(1), 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  PVector f = LVector({0.5, 0.7}).probability();
  BOOST_CHECK(vector_properties(f, 2));
  BOOST_CHECK_CLOSE(f(0), std::exp(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f(1), std::exp(0.7), 1e-8);

  PVector g = PTable({3}, {0.1, 0.2, 0.3}).vector();
  BOOST_CHECK(vector_properties(g, 3));
  BOOST_CHECK_EQUAL(g(0), 0.1);
  BOOST_CHECK_EQUAL(g(1), 0.2);
  BOOST_CHECK_EQUAL(g(2), 0.3);

  swap(f, g);
  BOOST_CHECK(vector_properties(f, 3));
  BOOST_CHECK(vector_properties(g, 2));
}

BOOST_AUTO_TEST_CASE(test_transform) {
  PVector f({1, 2, 3});
  PVector g({0.5, 1.0, 1.5});

  // Scalar transforms
  BOOST_CHECK_SMALL(max_diff(f * 2.0, PVector({2.0, 4.0, 6.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(2.0 * f, PVector({2.0, 4.0, 6.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / 2.0, PVector({0.5, 1.0, 1.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(6.0 / f, PVector({6.0, 3.0, 2.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), PVector({1.0, 4.0, 9.0})), 1e-8);

  // Vector transforms
  BOOST_CHECK_SMALL(max_diff(f * g, PVector({0.5, 2.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, PVector({2.0, 2.0, 2.0})), 1e-8);

  // Weighted update
  BOOST_CHECK_SMALL(max_diff(weighted_update(f, g, 0.3), PVector({0.85, 1.7, 2.55})), 1e-8);

  // Aggregates and normalization
  std::vector<size_t> argmax;
  std::vector<size_t> argmin;
  BOOST_CHECK_CLOSE(f.maximum(&argmax), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f.minimum(&argmin), 1.0, 1e-8);
  BOOST_CHECK_EQUAL(argmax.size(), 1);
  BOOST_CHECK_EQUAL(argmin.size(), 1);
  BOOST_CHECK_EQUAL(argmax[0], 2);
  BOOST_CHECK_EQUAL(argmin[0], 0);

  PVector n = f;
  n.normalize();
  BOOST_CHECK_CLOSE(n.marginal(), 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_access_assignment_and_conversion) {
  PVector s(Shape{3}, 2.5);
  BOOST_CHECK(vector_properties(s, 3));
  BOOST_CHECK_EQUAL(s(0), 2.5);
  BOOST_CHECK_EQUAL(s(1), 2.5);
  BOOST_CHECK_EQUAL(s(2), 2.5);

  PVector p(3);
  p.param() << 1.0, 2.0, 3.0;
  const auto& cp = p.param();
  BOOST_CHECK_EQUAL(cp.size(), 3);
  BOOST_CHECK_EQUAL(cp(0), 1.0);
  BOOST_CHECK_EQUAL(cp(1), 2.0);
  BOOST_CHECK_EQUAL(cp(2), 3.0);

  BOOST_CHECK_EQUAL(p(std::vector<size_t>{1}), 2.0);
  BOOST_CHECK_CLOSE(p.log(std::vector<size_t>{2}), std::log(3.0), 1e-8);

  PVector copy_ctor(p);
  BOOST_CHECK_SMALL(max_diff(copy_ctor, p), 1e-8);

  PVector copy_assign;
  copy_assign = p;
  BOOST_CHECK_SMALL(max_diff(copy_assign, p), 1e-8);

  PVector move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(max_diff(move_ctor, p), 1e-8);

  PVector move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(max_diff(move_assign, p), 1e-8);

  PTable t = p.table();
  BOOST_CHECK(table_properties(t, Shape{3}));
  BOOST_CHECK_EQUAL(t({0}), 1.0);
  BOOST_CHECK_EQUAL(t({1}), 2.0);
  BOOST_CHECK_EQUAL(t({2}), 3.0);
}

BOOST_AUTO_TEST_CASE(test_inplace_and_information) {
  PVector f({1.0, 2.0, 4.0});
  PVector g({0.5, 1.0, 2.0});

  PVector a = f;
  a *= 2.0;
  BOOST_CHECK_SMALL(max_diff(a, PVector({2.0, 4.0, 8.0})), 1e-8);
  a /= 2.0;
  BOOST_CHECK_SMALL(max_diff(a, f), 1e-8);

  a *= g;
  BOOST_CHECK_SMALL(max_diff(a, PVector({0.5, 2.0, 8.0})), 1e-8);
  a /= g;
  BOOST_CHECK_SMALL(max_diff(a, f), 1e-8);

  PVector p({0.2, 0.3, 0.5});
  PVector q({0.1, 0.4, 0.5});
  const double h = -(0.2 * std::log(0.2) + 0.3 * std::log(0.3) + 0.5 * std::log(0.5));
  const double ce = -(0.2 * std::log(0.1) + 0.3 * std::log(0.4) + 0.5 * std::log(0.5));
  const double kl = 0.2 * std::log(0.2 / 0.1) + 0.3 * std::log(0.3 / 0.4);

  BOOST_CHECK_CLOSE(p.entropy(), h, 1e-8);
  BOOST_CHECK_CLOSE(p.cross_entropy(q), ce, 1e-8);
  BOOST_CHECK_CLOSE(p.kl_divergence(q), kl, 1e-8);
  BOOST_CHECK_CLOSE(p.sum_diff(q), 0.2, 1e-8);
  BOOST_CHECK_CLOSE(p.max_diff(q), 0.1, 1e-8);
}
