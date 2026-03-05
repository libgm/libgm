#define BOOST_TEST_MODULE logarithmic_vector
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_vector.hpp>

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LVector = LogarithmicVector<double>;
using PVector = ProbabilityVector<double>;
using LTable = LogarithmicTable<double>;

BOOST_AUTO_TEST_CASE(test_constructors) {
  LVector a(3);
  BOOST_CHECK(vector_properties(a, 3));

  LVector b(2, Exp<double>(3.0));
  BOOST_CHECK(vector_properties(b, 2));
  BOOST_CHECK_EQUAL(b.log(0), 3.0);
  BOOST_CHECK_EQUAL(b.log(1), 3.0);

  LVector c(Eigen::Vector3d(2.0, 3.0, 4.0));
  BOOST_CHECK(vector_properties(c, 3));
  BOOST_CHECK_EQUAL(c.log(0), 2.0);
  BOOST_CHECK_EQUAL(c.log(1), 3.0);
  BOOST_CHECK_EQUAL(c.log(2), 4.0);

  LVector d = {6.0, 6.5};
  BOOST_CHECK(vector_properties(d, 2));
  BOOST_CHECK_EQUAL(d.log(0), 6.0);
  BOOST_CHECK_EQUAL(d.log(1), 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_swap) {
  LVector f = PVector({0.5, 0.7}).logarithmic();
  BOOST_CHECK(vector_properties(f, 2));
  BOOST_CHECK_CLOSE(f.log(0), std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f.log(1), std::log(0.7), 1e-8);

  LVector g = LTable({3}, {0.1, 0.2, 0.3}).vector();
  BOOST_CHECK(vector_properties(g, 3));
  BOOST_CHECK_EQUAL(g.log(0), 0.1);
  BOOST_CHECK_EQUAL(g.log(1), 0.2);
  BOOST_CHECK_EQUAL(g.log(2), 0.3);

  swap(f, g);
  BOOST_CHECK(vector_properties(f, 3));
  BOOST_CHECK(vector_properties(g, 2));
}

BOOST_AUTO_TEST_CASE(test_transform) {
  LVector f({1, 2, 3});
  LVector g({0.5, 1.0, 1.5});

  Exp<double> one(1.0);
  Exp<double> half(0.5);
  Exp<double> four(4.0);

  // Scalar transforms
  BOOST_CHECK_SMALL(max_diff(f * one, LVector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(one * f, LVector({2.0, 3.0, 4.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / half, LVector({0.5, 1.5, 2.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(four / f, LVector({3.0, 2.0, 1.0})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(pow(f, 2.0), LVector({2.0, 4.0, 6.0})), 1e-8);

  // Vector transforms
  BOOST_CHECK_SMALL(max_diff(f * g, LVector({1.5, 3.0, 4.5})), 1e-8);
  BOOST_CHECK_SMALL(max_diff(f / g, LVector({0.5, 1.0, 1.5})), 1e-8);

  // Weighted update
  BOOST_CHECK_SMALL(max_diff(weighted_update(f, g, 0.3), LVector({0.85, 1.7, 2.55})), 1e-8);

  // Aggregates
  std::vector<size_t> argmax;
  std::vector<size_t> argmin;
  BOOST_CHECK_CLOSE(log(f.maximum(&argmax)), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(log(f.minimum(&argmin)), 1.0, 1e-8);
  BOOST_CHECK_EQUAL(argmax.size(), 1);
  BOOST_CHECK_EQUAL(argmin.size(), 1);
  BOOST_CHECK_EQUAL(argmax[0], 2);
  BOOST_CHECK_EQUAL(argmin[0], 0);
}

BOOST_AUTO_TEST_CASE(test_access_assignment_and_conversion) {
  LVector s(Shape{3}, Exp<double>(1.5));
  BOOST_CHECK(vector_properties(s, 3));
  BOOST_CHECK_EQUAL(s.log(0), 1.5);
  BOOST_CHECK_EQUAL(s.log(1), 1.5);
  BOOST_CHECK_EQUAL(s.log(2), 1.5);

  LVector l(3);
  l.param() << 1.0, 2.0, 3.0;
  const auto& cl = l.param();
  BOOST_CHECK_EQUAL(cl.size(), 3);
  BOOST_CHECK_EQUAL(cl(0), 1.0);
  BOOST_CHECK_EQUAL(cl(1), 2.0);
  BOOST_CHECK_EQUAL(cl(2), 3.0);

  BOOST_CHECK_CLOSE(log(l(std::vector<size_t>{1})), 2.0, 1e-8);
  BOOST_CHECK_EQUAL(l.log(std::vector<size_t>{2}), 3.0);

  LVector copy_ctor(l);
  BOOST_CHECK_SMALL(max_diff(copy_ctor, l), 1e-8);

  LVector copy_assign;
  copy_assign = l;
  BOOST_CHECK_SMALL(max_diff(copy_assign, l), 1e-8);

  LVector move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(max_diff(move_ctor, l), 1e-8);

  LVector move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(max_diff(move_assign, l), 1e-8);

  LTable t = l.table();
  BOOST_CHECK(table_properties(t, Shape{3}));
  BOOST_CHECK_EQUAL(t.log({0}), 1.0);
  BOOST_CHECK_EQUAL(t.log({1}), 2.0);
  BOOST_CHECK_EQUAL(t.log({2}), 3.0);
}

BOOST_AUTO_TEST_CASE(test_inplace_and_information) {
  LVector f({1.0, 2.0, 4.0});
  LVector g({0.5, 1.0, 2.0});

  LVector a = f;
  a *= Exp<double>(2.0);
  BOOST_CHECK_SMALL(max_diff(a, LVector({3.0, 4.0, 6.0})), 1e-8);
  a /= Exp<double>(2.0);
  BOOST_CHECK_SMALL(max_diff(a, f), 1e-8);

  a *= g;
  BOOST_CHECK_SMALL(max_diff(a, LVector({1.5, 3.0, 6.0})), 1e-8);
  a /= g;
  BOOST_CHECK_SMALL(max_diff(a, f), 1e-8);

  LVector p({std::log(0.2), std::log(0.3), std::log(0.5)});
  LVector q({std::log(0.1), std::log(0.4), std::log(0.5)});
  const double h = -(0.2 * std::log(0.2) + 0.3 * std::log(0.3) + 0.5 * std::log(0.5));
  const double ce = -(0.2 * std::log(0.1) + 0.3 * std::log(0.4) + 0.5 * std::log(0.5));
  const double kl = 0.2 * std::log(0.2 / 0.1) + 0.3 * std::log(0.3 / 0.4);
  const double sd = std::abs(std::log(0.2) - std::log(0.1))
                  + std::abs(std::log(0.3) - std::log(0.4));
  const double md = std::max(std::abs(std::log(0.2) - std::log(0.1)),
                             std::abs(std::log(0.3) - std::log(0.4)));

  BOOST_CHECK_CLOSE(p.entropy(), h, 1e-8);
  BOOST_CHECK_CLOSE(p.cross_entropy(q), ce, 1e-8);
  BOOST_CHECK_CLOSE(p.kl_divergence(q), kl, 1e-8);
  BOOST_CHECK_CLOSE(p.sum_diff(q), sd, 1e-8);
  BOOST_CHECK_CLOSE(p.max_diff(q), md, 1e-8);
}
