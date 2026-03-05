#define BOOST_TEST_MODULE logarithmic_table
#include <boost/test/unit_test.hpp>

#include <libgm/factor/logarithmic_table.hpp>

#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_table.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LMatrix = LogarithmicMatrix<double>;
using LTable = LogarithmicTable<double>;
using LVector = LogarithmicVector<double>;
using PTable = ProbabilityTable<double>;

namespace {
Dims make_dims(std::initializer_list<size_t> idx) {
  Dims d;
  for (size_t i : idx) {
    d.set(i);
  }
  return d;
}
} // namespace

BOOST_AUTO_TEST_CASE(test_constructors) {
  LTable b({2, 3});
  BOOST_CHECK(table_properties(b, Shape{2, 3}));

  LTable c(Exp<double>(2.0));
  BOOST_CHECK(table_properties(c, Shape{}));
  BOOST_CHECK_EQUAL(c.param()[0], 2.0);

  LTable d({2}, Exp<double>(3.0));
  BOOST_CHECK(table_properties(d, Shape{2}));
  BOOST_CHECK_EQUAL(d.log({0}), 3.0);
  BOOST_CHECK_EQUAL(d.log({1}), 3.0);

  Table<double> params({2, 3}, 5.0);
  LTable f(params);
  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK_EQUAL(f.log({0, 0}), 5.0);
  BOOST_CHECK_EQUAL(f.log({1, 2}), 5.0);

  LTable g({2}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, Shape{2}));
  BOOST_CHECK_EQUAL(g.log({0}), 6.0);
  BOOST_CHECK_EQUAL(g.log({1}), 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_and_swap) {
  LTable f;
  f = LTable({2, 3});
  BOOST_CHECK(table_properties(f, Shape{2, 3}));

  f = LTable(Exp<double>(3.0));
  BOOST_CHECK(table_properties(f, Shape{}));
  BOOST_CHECK_EQUAL(f.param()[0], 3.0);

  PTable pt({2}, {0.5, 0.7});
  f = pt.logarithmic();
  BOOST_CHECK(table_properties(f, Shape{2}));
  BOOST_CHECK_CLOSE(f.log({0}), std::log(0.5), 1e-8);
  BOOST_CHECK_CLOSE(f.log({1}), std::log(0.7), 1e-8);

  LTable g({2, 3});
  swap(f, g);
  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK(table_properties(g, Shape{2}));
}

BOOST_AUTO_TEST_CASE(test_conversions) {
  LTable unary({2}, {0.25, 0.75});
  LVector v = unary.vector();
  BOOST_CHECK(vector_properties(v, 2));
  BOOST_CHECK_EQUAL(v.log(0), 0.25);
  BOOST_CHECK_EQUAL(v.log(1), 0.75);

  LTable binary({2, 3}, {1, 2, 3, 4, 5, 6});
  LMatrix m = binary.matrix();
  BOOST_CHECK(matrix_properties(m, 2, 3));
  BOOST_CHECK_EQUAL(m.log(0, 0), 1.0);
  BOOST_CHECK_EQUAL(m.log(1, 2), 6.0);

  PTable p = binary.probability();
  BOOST_CHECK(table_properties(p, Shape{2, 3}));
  BOOST_CHECK_CLOSE(p({0, 0}), std::exp(1.0), 1e-8);
  BOOST_CHECK_CLOSE(p({1, 2}), std::exp(6.0), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_copy_move_and_param) {
  LTable f({2, 2}, {1.0, 2.0, 3.0, 4.0});
  f.param()[1] = 7.0;
  const LTable& cf = f;
  BOOST_CHECK_EQUAL(cf.param()[1], 7.0);

  LTable copy_ctor(f);
  BOOST_CHECK_SMALL(copy_ctor.max_diff(f), 1e-8);

  LTable copy_assign;
  copy_assign = f;
  BOOST_CHECK_SMALL(copy_assign.max_diff(f), 1e-8);

  LTable move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(move_ctor.max_diff(f), 1e-8);

  LTable move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(move_assign.max_diff(f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_indexing) {
  LTable f({2, 3});
  std::iota(f.param().begin(), f.param().end(), 1.0);

  BOOST_CHECK_EQUAL(f.log({0, 0}), 1.0);
  BOOST_CHECK_EQUAL(f.log({1, 0}), 2.0);
  BOOST_CHECK_EQUAL(f.log({0, 1}), 3.0);
  BOOST_CHECK_EQUAL(f.log({1, 1}), 4.0);
  BOOST_CHECK_EQUAL(f.log({0, 2}), 5.0);
  BOOST_CHECK_EQUAL(f.log({1, 2}), 6.0);
  BOOST_CHECK_EQUAL(log(f({0, 2})), 5.0);
}

BOOST_AUTO_TEST_CASE(test_elementwise) {
  const double raw[] = {1, 2, 3, 4, 5, 6};
  LTable f({2, 3}, raw);
  LTable g({2, 3}, {2, 2, 2, 2, 2, 2});

  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK_EQUAL(f.log({0, 0}), 1.0);
  BOOST_CHECK_EQUAL(f.log({1, 2}), 6.0);

  LTable h = f * g;
  BOOST_CHECK(table_properties(h, Shape{2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) + 2.0, 1e-8);
    }
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, Shape{2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) - 2.0, 1e-8);
    }
  }

  h = f;
  h *= g;
  h /= g;
  BOOST_CHECK_CLOSE(h.max_diff(f), 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_scalar) {
  LTable f({2, 2}, {0, 1, 2, 3});
  LTable h;

  h = f * Exp<double>(2.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) + 2.0, 1e-8);
    }
  }

  h *= Exp<double>(1.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) + 3.0, 1e-8);
    }
  }

  h = Exp<double>(2.0) * f;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) + 2.0, 1e-8);
    }
  }

  h = Exp<double>(2.0) * f;
  h /= Exp<double>(1.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) + 1.0, 1e-8);
    }
  }

  h = f / Exp<double>(2.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), f.log({x, y}) - 2.0, 1e-8);
    }
  }

  h = Exp<double>(2.0) / f;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), 2.0 - f.log({x, y}), 1e-8);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_join_front_back) {
  LTable f({2, 3, 2});
  std::iota(f.param().begin(), f.param().end(), 1.0);
  LTable fx({2}, {10.0, 100.0});
  LTable fz({2}, {2.0, 3.0});

  LTable h = f.multiply_front(fx);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y, z}) + fx.log({x}), 1e-8);
      }
    }
  }

  h = f.multiply_back(fz);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y, z}) + fz.log({z}), 1e-8);
      }
    }
  }

  h = f.divide_front(fx);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y, z}) - fx.log({x}), 1e-8);
      }
    }
  }

  h = f.divide_back(fz);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y, z}) - fz.log({z}), 1e-8);
      }
    }
  }

  LTable in = f;
  in.multiply_in_front(fx);
  BOOST_CHECK_CLOSE(in.max_diff(f.multiply_front(fx)), 0.0, 1e-8);
  in = f;
  in.multiply_in_back(fz);
  BOOST_CHECK_CLOSE(in.max_diff(f.multiply_back(fz)), 0.0, 1e-8);
  in = f;
  in.divide_in_front(fx);
  BOOST_CHECK_CLOSE(in.max_diff(f.divide_front(fx)), 0.0, 1e-8);
  in = f;
  in.divide_in_back(fz);
  BOOST_CHECK_CLOSE(in.max_diff(f.divide_back(fz)), 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_join_dims) {
  LTable f({2, 2}, {0, 1, 2, 3});       // x, y
  LTable g({2, 3}, {1, 2, 3, 4, 5, 6}); // y, z
  LTable h;

  h = multiply(f, g, make_dims({0, 1}), make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y}) + g.log({y, z}), 1e-8);
      }
    }
  }

  h.multiply_in(g, make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y}) + 2 * g.log({y, z}), 1e-8);
      }
    }
  }

  h = divide(f, g, make_dims({0, 1}), make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), f.log({x, y}) - g.log({y, z}), 1e-8);
      }
    }
  }

  h.divide_in(f, make_dims({0, 1}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h.log({x, y, z}), -g.log({y, z}), 1e-8);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_arithmetic) {
  LTable f({2, 2}, {0, 1, 2, 3});
  LTable h = pow(f, 2.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h.log({x, y}), 2.0 * f.log({x, y}), 1e-8);
    }
  }

  LTable f1({2, 2}, {0, 1, 2, 3});
  LTable f2({2, 2}, {-2, 3, 0, 0});
  h = weighted_update(f1, f2, 0.3);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h.param()[i], 0.7 * f1.param()[i] + 0.3 * f2.param()[i], 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_maximum) {
  LTable f({2, 3, 2}, {
    10, 11,  // y=0, z=0
    50, 60,  // y=1, z=0
    20, 21,  // y=2, z=0
    12, 13,  // y=0, z=1
    80, 70,  // y=1, z=1
    22, 23   // y=2, z=1
  });

  LTable mfront = f.maximum_front(1);
  LTable mback = f.maximum_back(1);
  LTable mdims = f.maximum_dims(make_dims({0, 2}));

  BOOST_CHECK(table_properties(mfront, Shape{2}));
  BOOST_CHECK(table_properties(mback, Shape{2}));
  BOOST_CHECK(table_properties(mdims, Shape{2, 2}));

  for (size_t x = 0; x < 2; ++x) {
    double expected = -std::numeric_limits<double>::infinity();
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        expected = std::max(expected, f.log({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mfront.log({x}), expected);
  }

  for (size_t z = 0; z < 2; ++z) {
    double expected = -std::numeric_limits<double>::infinity();
    for (size_t x = 0; x < 2; ++x) {
      for (size_t y = 0; y < 3; ++y) {
        expected = std::max(expected, f.log({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mback.log({z}), expected);
  }

  for (size_t x = 0; x < 2; ++x) {
    for (size_t z = 0; z < 2; ++z) {
      double expected = -std::numeric_limits<double>::infinity();
      for (size_t y = 0; y < 3; ++y) {
        expected = std::max(expected, f.log({x, y, z}));
      }
      BOOST_CHECK_EQUAL(mdims.log({x, z}), expected);
    }
  }

  std::vector<size_t> vec;
  BOOST_CHECK_EQUAL(log(f.maximum(&vec)), 80.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 1);
  BOOST_CHECK_EQUAL(vec[2], 1);
}

BOOST_AUTO_TEST_CASE(test_minimum) {
  LTable f({2, 3, 2}, {
    40, 41,  // y=0, z=0
    5, 6,    // y=1, z=0
    30, 31,  // y=2, z=0
    42, 43,  // y=0, z=1
    1, 2,    // y=1, z=1
    32, 33   // y=2, z=1
  });

  LTable mfront = f.minimum_front(1);
  LTable mback = f.minimum_back(1);
  LTable mdims = f.minimum_dims(make_dims({0, 2}));

  BOOST_CHECK(table_properties(mfront, Shape{2}));
  BOOST_CHECK(table_properties(mback, Shape{2}));
  BOOST_CHECK(table_properties(mdims, Shape{2, 2}));

  for (size_t x = 0; x < 2; ++x) {
    double expected = std::numeric_limits<double>::infinity();
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        expected = std::min(expected, f.log({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mfront.log({x}), expected);
  }

  for (size_t z = 0; z < 2; ++z) {
    double expected = std::numeric_limits<double>::infinity();
    for (size_t x = 0; x < 2; ++x) {
      for (size_t y = 0; y < 3; ++y) {
        expected = std::min(expected, f.log({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mback.log({z}), expected);
  }

  for (size_t x = 0; x < 2; ++x) {
    for (size_t z = 0; z < 2; ++z) {
      double expected = std::numeric_limits<double>::infinity();
      for (size_t y = 0; y < 3; ++y) {
        expected = std::min(expected, f.log({x, y, z}));
      }
      BOOST_CHECK_EQUAL(mdims.log({x, z}), expected);
    }
  }

  std::vector<size_t> vec;
  BOOST_CHECK_EQUAL(log(f.minimum(&vec)), 1.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 1);
  BOOST_CHECK_EQUAL(vec[2], 1);
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  LTable f({2, 3}, {0, 1, 2, 3, 5, 6});
  LTable h;

  std::vector<double> fall2 = {5, 6};
  h = f.restrict_back({2});
  BOOST_CHECK(table_properties(h, Shape{2}));
  BOOST_CHECK(range_equal(h.param(), fall2));

  std::vector<double> f1all = {1, 3, 6};
  h = f.restrict_front({1});
  BOOST_CHECK(table_properties(h, Shape{3}));
  BOOST_CHECK(range_equal(h.param(), f1all));

  std::vector<double> f12 = {6};
  h = f.restrict_dims(make_dims({0, 1}), {1, 2});
  BOOST_CHECK(table_properties(h, Shape{}));
  BOOST_CHECK(range_equal(h.param(), f12));

  LTable f3({2, 3, 2});
  std::iota(f3.param().begin(), f3.param().end(), 1.0);
  h = f3.restrict_dims(make_dims({0, 2}), {1, 0});
  BOOST_CHECK(table_properties(h, Shape{3}));
  BOOST_CHECK_EQUAL(h.log({0}), f3.log({1, 0, 0}));
  BOOST_CHECK_EQUAL(h.log({1}), f3.log({1, 1, 0}));
  BOOST_CHECK_EQUAL(h.log({2}), f3.log({1, 2, 0}));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  PTable pp({2, 2}, {0.1, 0.2, 0.3, 0.4});
  PTable qp({2, 2}, {0.4 * 0.3, 0.6 * 0.3, 0.4 * 0.7, 0.6 * 0.7});
  LTable p = pp.logarithmic();
  LTable q = qp.logarithmic();

  const double hp = -(0.1 * log(0.1) + 0.2 * log(0.2) + 0.3 * log(0.3) + 0.4 * log(0.4));
  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      const double pv = pp({i, j});
      const double qv = qp({i, j});
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

BOOST_AUTO_TEST_CASE(test_edge_zero_behavior) {
  PTable pz({2, 2}, {0.0, 1.0, 2.0, 0.0});
  LTable z = pz.logarithmic();
  BOOST_CHECK(std::isinf(z.log({0, 0})));
  BOOST_CHECK(std::isinf(z.log({1, 1})));

  LTable inv = Exp<double>(3.0) / z;
  BOOST_CHECK(std::isinf(inv.log({0, 0})));
  BOOST_CHECK_CLOSE(inv.log({0, 1}), 3.0 - std::log(2.0), 1e-8);
  BOOST_CHECK_CLOSE(inv.log({1, 0}), 3.0, 1e-8);
  BOOST_CHECK(std::isinf(inv.log({1, 1})));
}
