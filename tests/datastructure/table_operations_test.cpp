#define BOOST_TEST_MODULE table_operations
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/table.hpp>
#include <libgm/datastructure/table_operations.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

using namespace libgm;

using IntTable = Table<int>;

namespace {

Dims make_dims(std::initializer_list<size_t> bits) {
  Dims d;
  for (size_t bit : bits) {
    d.set(bit);
  }
  return d;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_table_increment_select) {
  Shape shape3 = {2, 3, 2};
  TableIncrement inc0(shape3,make_dims({0}));
  BOOST_CHECK_EQUAL(inc0.size(), 3);
  BOOST_CHECK_EQUAL(inc0[0], 1);
  BOOST_CHECK_EQUAL(inc0[1], -1);
  BOOST_CHECK_EQUAL(inc0[2], -1);
  BOOST_CHECK_EQUAL(inc0[3], -1);

  TableIncrement inc1(shape3,make_dims({1}));
  BOOST_CHECK_EQUAL(inc1.size(), 3);
  BOOST_CHECK_EQUAL(inc1[0], 0);
  BOOST_CHECK_EQUAL(inc1[1], 1);
  BOOST_CHECK_EQUAL(inc1[2], -2);
  BOOST_CHECK_EQUAL(inc1[3], -2);

  TableIncrement inc2(shape3, make_dims({2}));
  BOOST_CHECK_EQUAL(inc2.size(), 3);
  BOOST_CHECK_EQUAL(inc2[0], 0);
  BOOST_CHECK_EQUAL(inc2[1], 0);
  BOOST_CHECK_EQUAL(inc2[2], 1);
  BOOST_CHECK_EQUAL(inc2[3], -1);

  TableIncrement inc01(shape3, make_dims({0, 1}));
  BOOST_CHECK_EQUAL(inc01.size(), 3);
  BOOST_CHECK_EQUAL(inc01[0], 1);
  BOOST_CHECK_EQUAL(inc01[1], 1);
  BOOST_CHECK_EQUAL(inc01[2], -5);
  BOOST_CHECK_EQUAL(inc01[3], -5);;

  TableIncrement inc02(shape3, make_dims({0, 2}));
  BOOST_CHECK_EQUAL(inc02.size(), 3);
  BOOST_CHECK_EQUAL(inc02[0], 1);
  BOOST_CHECK_EQUAL(inc02[1], -1);
  BOOST_CHECK_EQUAL(inc02[2], 1);
  BOOST_CHECK_EQUAL(inc02[3], -3);
}

BOOST_AUTO_TEST_CASE(test_table_increment_restrict) {
  // (0, 0, 0)
  // (1, 0, 0)
  // (0, 1, 0)
  // (1, 1, 0)
  // (0, 2, 0)
  // (1, 2, 0)
  // (0, 0, 1)
  // (1, 0, 1)
  // (0, 1, 1)
  // (1, 1, 1)
  // (0, 2, 1)
  // (1, 2, 1)

  Shape shape3 = {2, 3, 2};
  TableIncrement inc0(make_dims({0}), shape3);
  BOOST_CHECK_EQUAL(inc0.size(), 2);
  BOOST_CHECK_EQUAL(inc0[0], 2);
  BOOST_CHECK_EQUAL(inc0[1], 2);
  BOOST_CHECK_EQUAL(inc0[2], -10);

  TableIncrement inc1(make_dims({1}), shape3);
  BOOST_CHECK_EQUAL(inc1.size(), 2);
  BOOST_CHECK_EQUAL(inc1[0], 1);
  BOOST_CHECK_EQUAL(inc1[1], 5);
  BOOST_CHECK_EQUAL(inc1[2], -7);

  TableIncrement inc2(make_dims({2}), shape3);
  BOOST_CHECK_EQUAL(inc2.size(), 2);
  BOOST_CHECK_EQUAL(inc2[0], 1);
  BOOST_CHECK_EQUAL(inc2[1], 1);
  BOOST_CHECK_EQUAL(inc2[2], -5);

  Shape shape4(4, 2);
  TableIncrement inc01(make_dims({0, 1}), shape4);
  BOOST_CHECK_EQUAL(inc01.size(), 2);
  BOOST_CHECK_EQUAL(inc01[0], 4);
  BOOST_CHECK_EQUAL(inc01[1], 4);
  BOOST_CHECK_EQUAL(inc01[2], -12);

  TableIncrement inc02(make_dims({0, 2}), shape4);
  BOOST_CHECK_EQUAL(inc02.size(), 2);
  BOOST_CHECK_EQUAL(inc02[0], 2);
  BOOST_CHECK_EQUAL(inc02[1], 6);
  BOOST_CHECK_EQUAL(inc02[2], -10);
}

BOOST_AUTO_TEST_CASE(test_sequential_ops) {
  IntTable x({2, 2});
  IntTable y({2, 2});

  std::iota(x.begin(), x.end(), 2);
  transform_in(x, [](int v) { return v + 3; });
  BOOST_CHECK_EQUAL(x[0], 5);
  BOOST_CHECK_EQUAL(x[1], 6);
  BOOST_CHECK_EQUAL(x[2], 7);
  BOOST_CHECK_EQUAL(x[3], 8);

  IntTable r;
  std::iota(x.begin(), x.end(), 1);
  std::iota(y.begin(), y.end(), 3);
  transform(x, y, std::plus<int>(), r);
  BOOST_CHECK_EQUAL(r.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(r[0], 4);
  BOOST_CHECK_EQUAL(r[1], 6);
  BOOST_CHECK_EQUAL(r[2], 8);
  BOOST_CHECK_EQUAL(r[3], 10);

  IntTable s = r;
  transform_in(s, x, [](int a, int b) { return a * 2 - b; });
  BOOST_CHECK_EQUAL(s.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(s[0], 4 * 2 - 1);
  BOOST_CHECK_EQUAL(s[1], 6 * 2 - 2);
  BOOST_CHECK_EQUAL(s[2], 8 * 2 - 3);
  BOOST_CHECK_EQUAL(s[3], 10 * 2 - 4);

  int sum = std::accumulate(r.begin(), r.end(), 1, std::plus<int>());
  BOOST_CHECK_EQUAL(sum, 29);
}

BOOST_AUTO_TEST_CASE(test_transform_unary_out_of_place) {
  IntTable x({2, 3});
  std::iota(x.begin(), x.end(), 1); // 1..6

  IntTable y;
  transform(x, [](int v) { return v * v + 1; }, y);
  BOOST_CHECK_EQUAL(y.shape(), x.shape());

  for (size_t i = 0; i < x.size(); ++i) {
    BOOST_CHECK_EQUAL(y[i], x[i] * x[i] + 1);
  }
}

BOOST_AUTO_TEST_CASE(test_join) {
  const size_t m = 10;
  const size_t n = 8;
  const size_t o = 9;

  int xa[m][n];
  int ya[n][o];
  int za[m][o];
  IntTable x({m, n});
  IntTable y({n, o});
  IntTable z({m, o});

  int value = 0;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      x({i, j}) = xa[i][j] = value++;
    }
  }
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < o; ++k) {
      y({j, k}) = ya[j][k] = value++;
    }
  }
  for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < o; ++k) {
      z({i, k}) = za[i][k] = value++;
    }
  }

  int sum_xy[m * n * o];
  int sum_xyz[m * n * o];
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        sum_xy[i + j * m + k * m * n] = xa[i][j] + ya[j][k];
        sum_xyz[i + j * m + k * m * n] = xa[i][j] + ya[j][k] + za[i][k];
      }
    }
  }

  IntTable result;
  join(x, y, make_dims({0, 1}), make_dims({1, 2}), std::plus<int>(), result);
  BOOST_CHECK_EQUAL(result.shape(), Shape({m, n, o}));
  BOOST_CHECK(std::equal(result.begin(), result.end(), sum_xy));

  join_in(result, z, make_dims({0, 2}), std::plus<int>());
  BOOST_CHECK_EQUAL(result.shape(), Shape({m, n, o}));
  BOOST_CHECK(std::equal(result.begin(), result.end(), sum_xyz));
}

BOOST_AUTO_TEST_CASE(test_join_front_back_variants) {
  const size_t m = 3;
  const size_t n = 2;
  const size_t o = 4;

  IntTable a({m, n, o});
  IntTable b_front({m, n}); // prefix of a
  IntTable b_back({n, o});  // suffix of a

  int value = 1;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        a({i, j, k}) = value++;
      }
    }
  }
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      b_front({i, j}) = value++;
    }
  }
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < o; ++k) {
      b_back({j, k}) = value++;
    }
  }

  // Manual expected: front join uses b_front(i,j), independent of k.
  std::vector<int> expected_front(a.size());
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        expected_front[i + j * m + k * m * n] = a({i, j, k}) + b_front({i, j});
      }
    }
  }

  // Manual expected: back join uses b_back(j,k), independent of i.
  std::vector<int> expected_back(a.size());
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        expected_back[i + j * m + k * m * n] = a({i, j, k}) + b_back({j, k});
      }
    }
  }

  IntTable out_front;
  join_front(a, b_front, std::plus<int>(), out_front);
  BOOST_CHECK_EQUAL(out_front.shape(), a.shape());
  BOOST_CHECK(std::equal(out_front.begin(), out_front.end(), expected_front.begin()));

  IntTable out_back;
  join_back(a, b_back, std::plus<int>(), out_back);
  BOOST_CHECK_EQUAL(out_back.shape(), a.shape());
  BOOST_CHECK(std::equal(out_back.begin(), out_back.end(), expected_back.begin()));

  IntTable in_front = a;
  join_in_front(in_front, b_front, std::plus<int>());
  BOOST_CHECK(std::equal(in_front.begin(), in_front.end(), expected_front.begin()));

  IntTable in_back = a;
  join_in_back(in_back, b_back, std::plus<int>());
  BOOST_CHECK(std::equal(in_back.begin(), in_back.end(), expected_back.begin()));
}

BOOST_AUTO_TEST_CASE(test_aggregate) {
  const size_t m = 10;
  const size_t n = 8;
  const size_t o = 9;

  int xa[m][n][o];
  IntTable x({m, n, o});

  int value = 2;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        x({i, j, k}) = xa[i][j][k] = value++;
      }
    }
  }

  int sum_mo[m * o];
  for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < o; ++k) {
      int tmp = 0;
      for (size_t j = 0; j < n; ++j) {
        tmp += xa[i][j][k];
      }
      sum_mo[i + k * m] = tmp;
    }
  }

  IntTable result_dims;
  aggregate(x, make_dims({0, 2}), 0, std::plus<int>(), result_dims);
  BOOST_CHECK_EQUAL(result_dims.shape(), Shape({m, o}));
  BOOST_CHECK(std::equal(result_dims.begin(), result_dims.end(), sum_mo));

  // Retain front dims {m,n}, aggregate over trailing dim k.
  int sum_mn[m * n];
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      int tmp = 0;
      for (size_t k = 0; k < o; ++k) {
        tmp += xa[i][j][k];
      }
      sum_mn[i + j * m] = tmp;
    }
  }

  IntTable result_front;
  aggregate_front(x, 2, 0, std::plus<int>(), result_front);
  BOOST_CHECK_EQUAL(result_front.shape(), Shape({m, n}));
  BOOST_CHECK(std::equal(result_front.begin(), result_front.end(), sum_mn));

  int sum_no[n * o];
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < o; ++k) {
      int tmp = 0;
      for (size_t i = 0; i < m; ++i) {
        tmp += xa[i][j][k];
      }
      sum_no[j + k * n] = tmp;
    }
  }

  IntTable result_back;
  aggregate_back(x, 2, 0, std::plus<int>(), result_back);
  BOOST_CHECK_EQUAL(result_back.shape(), Shape({n, o}));
  BOOST_CHECK(std::equal(result_back.begin(), result_back.end(), sum_no));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  const size_t m = 10;
  const size_t n = 8;
  const size_t o = 9;

  int xa[m][n][o];
  IntTable x({m, n, o});

  int value = 2;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < o; ++k) {
        x({i, j, k}) = xa[i][j][k] = value++;
      }
    }
  }

  IntTable result_dims;
  IntTable result_front;

  int rm[n * o];
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < o; ++k) {
      rm[j + k * n] = xa[3][j][k];
    }
  }
  restrict_front(x, std::vector<size_t>{3}, result_front);
  restrict(x, make_dims({0}), std::vector<size_t>{3}, result_dims);
  BOOST_CHECK_EQUAL(result_front.shape(), Shape({n, o}));
  BOOST_CHECK(std::equal(result_front.begin(), result_front.end(), rm));
  BOOST_CHECK_EQUAL(result_dims.shape(), Shape({n, o}));
  BOOST_CHECK(std::equal(result_dims.begin(), result_dims.end(), rm));

  int rj[n];
  for (size_t j = 0; j < n; ++j) {
    rj[j] = xa[5][j][7];
  }
  restrict(x, make_dims({0, 2}), std::vector<size_t>{5, 7}, result_dims);
  BOOST_CHECK_EQUAL(result_dims.shape(), Shape({n}));
  BOOST_CHECK(std::equal(result_dims.begin(), result_dims.end(), rj));

  int rmn[m * n];
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      rmn[i + j * m] = xa[i][j][6];
    }
  }
  IntTable result_back;
  restrict_back(x, std::vector<size_t>{6}, result_back);
  BOOST_CHECK_EQUAL(result_back.shape(), Shape({m, n}));
  BOOST_CHECK(std::equal(result_back.begin(), result_back.end(), rmn));
}
