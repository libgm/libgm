#define BOOST_TEST_MODULE probability_table
#include <boost/test/unit_test.hpp>

#include <libgm/factor/probability_table.hpp>

#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_vector.hpp>
#include <libgm/factor/probability_matrix.hpp>

#include "predicates.hpp"
#include <utility>

using namespace libgm;

using LTable = LogarithmicTable<double>;
using PTable = ProbabilityTable<double>;
using PVector = ProbabilityVector<double>;

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
  PTable b({2, 3});
  BOOST_CHECK(table_properties(b, Shape{2, 3}));

  PTable c(2.0);
  BOOST_CHECK(table_properties(c, Shape{}));
  BOOST_CHECK_EQUAL(c.param()[0], 2.0);

  PTable d({2}, 3.0);
  BOOST_CHECK(table_properties(d, Shape{2}));
  BOOST_CHECK_EQUAL(d({0}), 3.0);
  BOOST_CHECK_EQUAL(d({1}), 3.0);

  Table<double> params({2, 3}, 5.0);
  PTable f(params);
  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK_EQUAL(f({0, 0}), 5.0);
  BOOST_CHECK_EQUAL(f({1, 2}), 5.0);

  PTable g({2}, {6.0, 6.5});
  BOOST_CHECK(table_properties(g, Shape{2}));
  BOOST_CHECK_EQUAL(g({0}), 6.0);
  BOOST_CHECK_EQUAL(g({1}), 6.5);
}

BOOST_AUTO_TEST_CASE(test_assignment_and_swap) {
  PTable f;
  f = PTable({2, 3});
  BOOST_CHECK(table_properties(f, Shape{2, 3}));

  f = PTable(3.0);
  BOOST_CHECK(table_properties(f, Shape{}));
  BOOST_CHECK_EQUAL(f.param()[0], 3.0);

  LTable ct({2}, Exp<double>(std::log(0.5)));
  f = ct.probability();
  BOOST_CHECK(table_properties(f, Shape{2}));
  BOOST_CHECK_CLOSE(f({0}), 0.5, 1e-8);
  BOOST_CHECK_CLOSE(f({1}), 0.5, 1e-8);

  PTable g({2, 3});
  swap(f, g);
  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK(table_properties(g, Shape{2}));
}

BOOST_AUTO_TEST_CASE(test_conversions) {
  PTable unary({2}, {0.25, 0.75});
  PVector v = unary.vector();
  BOOST_CHECK(vector_properties(v, 2));
  BOOST_CHECK_CLOSE(v(0), 0.25, 1e-8);
  BOOST_CHECK_CLOSE(v(1), 0.75, 1e-8);

  PTable binary({2, 3}, {1, 2, 3, 4, 5, 6});
  auto m = binary.matrix();
  BOOST_CHECK(matrix_properties(m, 2, 3));
  BOOST_CHECK_EQUAL(m(0, 0), 1.0);
  BOOST_CHECK_EQUAL(m(1, 2), 6.0);

  LTable lf = binary.logarithmic();
  BOOST_CHECK_CLOSE(lf.log({0, 0}), std::log(binary({0, 0})), 1e-8);
  BOOST_CHECK_CLOSE(lf.log({1, 2}), std::log(binary({1, 2})), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_copy_move_and_param) {
  PTable f({2, 2}, {1.0, 2.0, 3.0, 4.0});
  f.param()[1] = 7.0;
  const PTable& cf = f;
  BOOST_CHECK_EQUAL(cf.param()[1], 7.0);

  PTable copy_ctor(f);
  BOOST_CHECK_SMALL(copy_ctor.max_diff(f), 1e-8);

  PTable copy_assign;
  copy_assign = f;
  BOOST_CHECK_SMALL(copy_assign.max_diff(f), 1e-8);

  PTable move_ctor(std::move(copy_ctor));
  BOOST_CHECK_SMALL(move_ctor.max_diff(f), 1e-8);

  PTable move_assign;
  move_assign = std::move(copy_assign);
  BOOST_CHECK_SMALL(move_assign.max_diff(f), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_indexing) {
  PTable f({2, 3});
  auto& p = f.param();
  std::iota(p.begin(), p.end(), 1.0);

  BOOST_CHECK_CLOSE(f({0, 0}), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1, 0}), 2.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0, 1}), 3.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1, 1}), 4.0, 1e-8);
  BOOST_CHECK_CLOSE(f({0, 2}), 5.0, 1e-8);
  BOOST_CHECK_CLOSE(f({1, 2}), 6.0, 1e-8);

  BOOST_CHECK_CLOSE(f.log({0, 2}), std::log(5.0), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_elementwise) {
  const double raw[] = {1, 2, 3, 4, 5, 6};
  PTable f({2, 3}, raw);
  PTable g({2, 3}, {2, 2, 2, 2, 2, 2});

  BOOST_CHECK(table_properties(f, Shape{2, 3}));
  BOOST_CHECK_EQUAL(f({0, 0}), 1.0);
  BOOST_CHECK_EQUAL(f({1, 2}), 6.0);

  PTable h = f * g;
  BOOST_CHECK(table_properties(h, Shape{2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), 2.0 * f({x, y}), 1e-8);
    }
  }

  h = f / g;
  BOOST_CHECK(table_properties(h, Shape{2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), 0.5 * f({x, y}), 1e-8);
    }
  }

  h = f;
  h *= g;
  h /= g;
  BOOST_CHECK_CLOSE(h.max_diff(f), 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_scalar) {
  PTable f({2, 2}, {0, 1, 2, 3});
  PTable h;

  h = f * 2.0;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), f({x, y}) * 2.0, 1e-8);
    }
  }

  h *= 3.0;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), f({x, y}) * 6.0, 1e-8);
    }
  }

  h = 2.0 * f;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), f({x, y}) * 2.0, 1e-8);
    }
  }

  h = 2.0 * f;
  h /= 4.0;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), f({x, y}) * 0.5, 1e-8);
    }
  }

  h = f / 3.0;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), f({x, y}) / 3.0, 1e-8);
    }
  }

  h = 3.0 / f;
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      if (f({x, y})) {
        BOOST_CHECK_CLOSE(h({x, y}), 3.0 / f({x, y}), 1e-8);
      } else {
        BOOST_CHECK(std::isinf(h({x, y})));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_join_front_back) {
  PTable f({2, 3, 2});
  std::iota(f.param().begin(), f.param().end(), 1.0);
  PTable fx({2}, {10.0, 100.0});
  PTable fz({2}, {2.0, 3.0});

  PTable h = f.multiply_front(fx);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y, z}) * fx({x}), 1e-8);
      }
    }
  }

  h = f.multiply_back(fz);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y, z}) * fz({z}), 1e-8);
      }
    }
  }

  h = f.divide_front(fx);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y, z}) / fx({x}), 1e-8);
      }
    }
  }

  h = f.divide_back(fz);
  BOOST_CHECK(table_properties(h, Shape{2, 3, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y, z}) / fz({z}), 1e-8);
      }
    }
  }

  PTable in = f;
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
  PTable f({2, 2}, {0, 1, 2, 3});       // x, y
  PTable g({2, 3}, {1, 2, 3, 4, 5, 6}); // y, z
  PTable h;

  h = multiply(f, g, make_dims({0, 1}), make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y}) * g({y, z}), 1e-8);
      }
    }
  }

  h.multiply_in(g, make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y}) * std::pow(g({y, z}), 2), 1e-8);
      }
    }
  }

  h = divide(f, g, make_dims({0, 1}), make_dims({1, 2}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        BOOST_CHECK_CLOSE(h({x, y, z}), f({x, y}) / g({y, z}), 1e-8);
      }
    }
  }

  h.divide_in(f, make_dims({0, 1}));
  BOOST_CHECK(table_properties(h, Shape{2, 2, 3}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      for (size_t z = 0; z < 3; ++z) {
        if (f({x, y})) {
          BOOST_CHECK_CLOSE(h({x, y, z}), 1.0 / g({y, z}), 1e-8);
        } else {
          BOOST_CHECK(std::isnan(h({x, y, z})));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_arithmetic) {
  PTable f({2, 2}, {0, 1, 2, 3});
  PTable h = pow(f, 3.0);
  BOOST_CHECK(table_properties(h, Shape{2, 2}));
  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      BOOST_CHECK_CLOSE(h({x, y}), std::pow(f({x, y}), 3.0), 1e-8);
    }
  }

  PTable f1({2, 2}, {0, 1, 2, 3});
  PTable f2({2, 2}, {-2, 3, 0, 0});
  h = weighted_update(f1, f2, 0.3);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(h.param()[i], 0.7 * f1.param()[i] + 0.3 * f2.param()[i], 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(test_marginal) {
  PTable f({2, 3, 2});
  std::iota(f.param().begin(), f.param().end(), 1.0);

  PTable mfront = f.marginal_front(1);
  PTable mback = f.marginal_back(1);
  PTable mdims = f.marginal_dims(make_dims({0, 2}));

  BOOST_CHECK(table_properties(mfront, Shape{2}));
  BOOST_CHECK(table_properties(mback, Shape{2}));
  BOOST_CHECK(table_properties(mdims, Shape{2, 2}));

  for (size_t x = 0; x < 2; ++x) {
    double expected = 0.0;
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        expected += f({x, y, z});
      }
    }
    BOOST_CHECK_CLOSE(mfront({x}), expected, 1e-8);
  }

  for (size_t z = 0; z < 2; ++z) {
    double expected = 0.0;
    for (size_t x = 0; x < 2; ++x) {
      for (size_t y = 0; y < 3; ++y) {
        expected += f({x, y, z});
      }
    }
    BOOST_CHECK_CLOSE(mback({z}), expected, 1e-8);
  }

  for (size_t x = 0; x < 2; ++x) {
    for (size_t z = 0; z < 2; ++z) {
      double expected = 0.0;
      for (size_t y = 0; y < 3; ++y) {
        expected += f({x, y, z});
      }
      BOOST_CHECK_CLOSE(mdims({x, z}), expected, 1e-8);
    }
  }

  BOOST_CHECK_CLOSE(f.marginal(), std::accumulate(f.param().begin(), f.param().end(), 0.0), 1e-8);
}

BOOST_AUTO_TEST_CASE(test_maximum) {
  // Layout is (x, y, z) with x varying fastest in the initializer list.
  // Values are chosen so maxima are interior, not first/last elements.
  PTable f({2, 3, 2}, {
    10, 11,  // y=0, z=0
    50, 60,  // y=1, z=0
    20, 21,  // y=2, z=0
    12, 13,  // y=0, z=1
    80, 70,  // y=1, z=1
    22, 23   // y=2, z=1
  });

  PTable mfront = f.maximum_front(1);
  PTable mback = f.maximum_back(1);
  PTable mdims = f.maximum_dims(make_dims({0, 2}));

  BOOST_CHECK(table_properties(mfront, Shape{2}));
  BOOST_CHECK(table_properties(mback, Shape{2}));
  BOOST_CHECK(table_properties(mdims, Shape{2, 2}));

  for (size_t x = 0; x < 2; ++x) {
    double expected = -std::numeric_limits<double>::infinity();
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        expected = std::max(expected, f({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mfront({x}), expected);
  }

  for (size_t z = 0; z < 2; ++z) {
    double expected = -std::numeric_limits<double>::infinity();
    for (size_t x = 0; x < 2; ++x) {
      for (size_t y = 0; y < 3; ++y) {
        expected = std::max(expected, f({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mback({z}), expected);
  }

  for (size_t x = 0; x < 2; ++x) {
    for (size_t z = 0; z < 2; ++z) {
      double expected = -std::numeric_limits<double>::infinity();
      for (size_t y = 0; y < 3; ++y) {
        expected = std::max(expected, f({x, y, z}));
      }
      BOOST_CHECK_EQUAL(mdims({x, z}), expected);
    }
  }

  std::vector<size_t> vec;
  BOOST_CHECK_EQUAL(f.maximum(&vec), 80.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 1);
  BOOST_CHECK_EQUAL(vec[2], 1);
}

BOOST_AUTO_TEST_CASE(test_minimum) {
  // Values are chosen so minima are interior, not first/last elements.
  PTable f({2, 3, 2}, {
    40, 41,  // y=0, z=0
    5, 6,    // y=1, z=0
    30, 31,  // y=2, z=0
    42, 43,  // y=0, z=1
    1, 2,    // y=1, z=1
    32, 33   // y=2, z=1
  });

  PTable mfront = f.minimum_front(1);
  PTable mback = f.minimum_back(1);
  PTable mdims = f.minimum_dims(make_dims({0, 2}));

  BOOST_CHECK(table_properties(mfront, Shape{2}));
  BOOST_CHECK(table_properties(mback, Shape{2}));
  BOOST_CHECK(table_properties(mdims, Shape{2, 2}));

  for (size_t x = 0; x < 2; ++x) {
    double expected = std::numeric_limits<double>::infinity();
    for (size_t y = 0; y < 3; ++y) {
      for (size_t z = 0; z < 2; ++z) {
        expected = std::min(expected, f({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mfront({x}), expected);
  }

  for (size_t z = 0; z < 2; ++z) {
    double expected = std::numeric_limits<double>::infinity();
    for (size_t x = 0; x < 2; ++x) {
      for (size_t y = 0; y < 3; ++y) {
        expected = std::min(expected, f({x, y, z}));
      }
    }
    BOOST_CHECK_EQUAL(mback({z}), expected);
  }

  for (size_t x = 0; x < 2; ++x) {
    for (size_t z = 0; z < 2; ++z) {
      double expected = std::numeric_limits<double>::infinity();
      for (size_t y = 0; y < 3; ++y) {
        expected = std::min(expected, f({x, y, z}));
      }
      BOOST_CHECK_EQUAL(mdims({x, z}), expected);
    }
  }

  std::vector<size_t> vec;
  BOOST_CHECK_EQUAL(f.minimum(&vec), 1.0);
  BOOST_CHECK_EQUAL(vec[0], 0);
  BOOST_CHECK_EQUAL(vec[1], 1);
  BOOST_CHECK_EQUAL(vec[2], 1);
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  PTable f({2, 3}, {0, 1, 2, 3, 5, 6});
  PTable h;

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

  PTable f3({2, 3, 2});
  std::iota(f3.param().begin(), f3.param().end(), 1.0);
  h = f3.restrict_dims(make_dims({0, 2}), {1, 0});
  BOOST_CHECK(table_properties(h, Shape{3}));
  BOOST_CHECK_EQUAL(h({0}), f3({1, 0, 0}));
  BOOST_CHECK_EQUAL(h({1}), f3({1, 1, 0}));
  BOOST_CHECK_EQUAL(h({2}), f3({1, 2, 0}));
}

BOOST_AUTO_TEST_CASE(test_entropy) {
  using std::log;
  PTable p({2, 2}, {0.1, 0.2, 0.3, 0.4});
  PTable q({2, 2}, {0.4 * 0.3, 0.6 * 0.3, 0.4 * 0.7, 0.6 * 0.7});
  PTable m = weighted_update(p, q, 0.5);

  const double hpxy = -(0.1 * log(0.1) + 0.2 * log(0.2) + 0.3 * log(0.3) + 0.4 * log(0.4));
  const double hpx = -(0.4 * log(0.4) + 0.6 * log(0.6));
  const double hpy = -(0.3 * log(0.3) + 0.7 * log(0.7));

  double hpq = 0.0, klpq = 0.0, sumdiff = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      const double pv = p({i, j});
      const double qv = q({i, j});
      hpq += -pv * log(qv);
      klpq += pv * log(pv / qv);
      const double diff = std::abs(pv - qv);
      sumdiff += diff;
      maxdiff = std::max(maxdiff, diff);
    }
  }

  BOOST_CHECK_CLOSE(p.entropy(), hpxy, 1e-6);
  BOOST_CHECK_CLOSE(p.marginal_front(1).entropy(), hpx, 1e-6);
  BOOST_CHECK_CLOSE(p.marginal_back(1).entropy(), hpy, 1e-6);
  BOOST_CHECK_CLOSE(p.cross_entropy(q), hpq, 1e-6);
  BOOST_CHECK_CLOSE(p.kl_divergence(q), klpq, 1e-6);
  BOOST_CHECK_CLOSE(sum_diff(p, q), sumdiff, 1e-6);
  BOOST_CHECK_CLOSE(max_diff(p, q), maxdiff, 1e-6);
}
