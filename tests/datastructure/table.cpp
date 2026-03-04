#define BOOST_TEST_MODULE table
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/table.hpp>

#include <functional>

namespace libgm {
  template class table<double>;
  template class table<float>;
  template class table<int>;
}

using namespace libgm;

typedef table<int> int_table;
BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);

BOOST_AUTO_TEST_CASE(test_accessors) {
  const std::size_t n = 10;
  const std::size_t d = 3;
  const std::size_t m = std::size_t(pow(d, n));

  // Constructor
  int_table x(uint_vector(n, d));
  BOOST_CHECK_EQUAL(x.size(), m);

  // Index iterator and indexing
  int value = 0;
  for (const uint_vector& index : x.indices()) {
    x(index) = value++;
  }
  std::vector<int> seq(m);
  std::iota(seq.begin(), seq.end(), 0);
  BOOST_CHECK(std::equal(seq.begin(), seq.end(), x.begin()));

  // Comparison operator
  int_table y = x;
  BOOST_CHECK_EQUAL(x, y);
  y[0] = 20;
  BOOST_CHECK_NE(x, y);
}


BOOST_AUTO_TEST_CASE(test_sequential) {
  int_table x({2, 2});
  int_table y({2, 2});

  // Fill
  x.fill(3);
  BOOST_CHECK_EQUAL(std::count(x.begin(), x.end(), 3), 4);

  // Unary in-place transform
  incremented_by<int> inc_op(3);
  std::iota(x.begin(), x.end(), 2);
  x.transform(inc_op);
  BOOST_CHECK_EQUAL(x[0], 5);
  BOOST_CHECK_EQUAL(x[1], 6);
  BOOST_CHECK_EQUAL(x[2], 7);
  BOOST_CHECK_EQUAL(x[3], 8);

  // Binary transform
  int_table r;
  std::plus<int> plus_op;
  std::iota(x.begin(), x.end(), 1);
  std::iota(y.begin(), y.end(), 3);
  table_transform_assign<int, std::plus<int> >(r, plus_op)(x, y);
  BOOST_CHECK_EQUAL(r.shape(), uint_vector({2, 2}));
  BOOST_CHECK_EQUAL(r[0], 4);
  BOOST_CHECK_EQUAL(r[1], 6);
  BOOST_CHECK_EQUAL(r[2], 8);
  BOOST_CHECK_EQUAL(r[3], 10);

  // Binary in place transform
  int_table s = r;
  std::minus<int> minus_op;
  table_transform_update<int, std::minus<int>, std::plus<int> >(s, minus_op)(s, x);
  BOOST_CHECK_EQUAL(s.shape(), uint_vector({2, 2}));
  BOOST_CHECK_EQUAL(s[0], 4*2 - 1);
  BOOST_CHECK_EQUAL(s[1], 6*2 - 2);
  BOOST_CHECK_EQUAL(s[2], 8*2 - 3);
  BOOST_CHECK_EQUAL(s[3], 10*2 - 4);

  // Accumulate
  BOOST_CHECK_EQUAL(r.accumulate(1, std::plus<int>()), 29);

  // Transform-accumulate
  std::iota(x.begin(), x.end(), 2);
  int sum = transform_accumulate(inc_op, plus_op, 0, x);
  BOOST_CHECK_EQUAL(sum, 26);
}


bool is_close(const table<double>& p, double v0, double v1) {
  return std::abs(p[0] - v0) < 1e-8 && std::abs(p[1] - v1) < 1e-8;
}


BOOST_AUTO_TEST_CASE(test_opt_vector) {
  const table<double> p({1, 2}, {1, 2});
  const table<double> q({1, 2}, {1.5, -0.5});
  table<double> r;

  r = p; r += q;
  BOOST_CHECK(is_close(r, 2.5, 1.5));

  r = p; r -= q;
  BOOST_CHECK(is_close(r, -0.5, 2.5));

  r = p; r += 1.0;
  BOOST_CHECK(is_close(r, 2, 3));

  r = p; r -= 1.0;
  BOOST_CHECK(is_close(r, 0, 1));

  r = p; r *= 2.0;
  BOOST_CHECK(is_close(r, 2, 4));

  r = p; r /= 2.0;
  BOOST_CHECK(is_close(r, 0.5, 1));

  r = p; axpy(2.0, q, r);
  BOOST_CHECK(is_close(r, 4, 1));

  BOOST_CHECK_CLOSE(dot(p, q), 0.5, 1e-8);
}


BOOST_AUTO_TEST_CASE(test_join) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n];
  int ya[n][o];
  int za[o][m];
  int_table x({m, n});
  int_table y({n, o});
  int_table z({o, m});

  // Initialize the input arrays and tables
  int value = 0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      x({i,j}) = xa[i][j] = value++;
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      y({j,k}) = ya[j][k] = value++;
    }
  }
  for (std::size_t k = 0; k < o; ++k) {
    for (std::size_t i = 0; i < m; ++i) {
      z({k,i}) = za[k][i] = value++;
    }
  }

  // Compute the sum using native arrays
  int sum_xy[m * n * o];
  int sum_xyz[m * n * o];
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < o; ++k) {
        sum_xy[i + j*m + k*m*n] = xa[i][j] + ya[j][k];
        sum_xyz[i + j*m + k*m*n] = xa[i][j] + ya[j][k] + za[k][i];
      }
    }
  }

  // Compute the join using block indexing
  int_table result_span;
  join(std::plus<int>(), x, y, span(1, 2), result_span);
  BOOST_CHECK_EQUAL(result_span.shape(), uint_vector({m, n, o}));
  BOOST_CHECK(std::equal(result_span.begin(), result_span.end(), sum_xy));

  // Compute the join using subset indexing
  int_table result_iref;
  join(std::plus<int>(), x, y, ivec{1, 2}, result_iref);
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({m, n, o}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), sum_xy));

  // Compute generic the inplace join
  z.join_inplace(std::plus<int>(), ivec{2, 0}, result_iref);
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({m, n, o}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), sum_xyz));
}


BOOST_AUTO_TEST_CASE(test_aggregate) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n][o];
  int_table x({m, n, o});

  // Initialize the input array and table
  int value = 2;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < o; ++k) {
        x({i,j,k}) = xa[i][j][k] = value++;
      }
    }
  }

  // Performs the aggreate operation using native arrays
  int sum[o*m];
  for (std::size_t k = 0; k < o; ++k) {
    for (std::size_t i = 0; i < m; ++i) {
      int tmp = 0;
      for (std::size_t j = 0; j < n; ++j) { tmp += xa[i][j][k]; }
      sum[k + i*o] = tmp;
    }
  }

  // Compute the aggregate using subset indexing
  int_table result_iref;
  x.aggregate(std::plus<int>(), 0, ivec{2, 0}, result_iref);
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({o, m}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), sum));

  // Compute the aggregate using span indexing
  int_table result_span;
  int_table reordered;
  x.reorder(ivec{1, 2, 0}, reordered);
  reordered.aggregate(std::plus<int>(), 0, span(1, 2), result_span);
  BOOST_CHECK_EQUAL(result_span.shape(), uint_vector({o, m}));
  BOOST_CHECK(std::equal(result_span.begin(), result_span.end(), sum));
}


BOOST_AUTO_TEST_CASE(test_join_aggregate) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n];
  int ya[n][o];
  int_table x({m, n});
  int_table y({n, o});

  // Initialize the input arrays and tables
  int value = 0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      x({i,j}) = xa[i][j] = value++;
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      y({j,k}) = ya[j][k] = value++;
    }
  }

  // Performs the join-aggregate operation using native arrays
  int sum[o*m];
  for (std::size_t k = 0; k < o; ++k) {
    for (std::size_t i = 0; i < m; ++i) {
      int tmp = 0;
      for (std::size_t j = 0; j < n; ++j) { tmp += xa[i][j] * ya[j][k]; }
      sum[k + i*o] = tmp;
    }
  }

  // Perform a generic join-aggregate
  std::multiplies<int> mult_op;
  std::plus<int> plus_op;
  int_table result;
  join_aggregate(mult_op, plus_op, 0, x, y, ivec{1, 2}, ivec{2, 0}, result);
  BOOST_CHECK_EQUAL(result.shape(), uint_vector({o, m}));
  BOOST_CHECK(std::equal(result.begin(), result.end(), sum));

  // Perform a generic join-accumulate
  int acc = join_accumulate(mult_op, plus_op, 0, x, y, ivec{1, 2});
  BOOST_CHECK_EQUAL(acc, result.accumulate(0, plus_op));
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n][o];
  int_table x({m, n, o});

  // Initialize the input array and table
  int value = 2;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < o; ++k) {
        x({i,j,k}) = xa[i][j][k] = value++;
      }
    }
  }

  int_table result_iref;
  int_table result_span;

  // Test prefix restrict
  int rm[n*o];
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      rm[j + k*n] = xa[3][j][k];
    }
  }
  x.restrict(front(1), uint_vector{3}, result_span);
  x.restrict(ivec{0}, uint_vector{3}, result_iref);
  BOOST_CHECK_EQUAL(result_span.shape(), uint_vector({n, o}));
  BOOST_CHECK(std::equal(result_span.begin(), result_span.end(), rm));
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({n, o}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), rm));

  // Test middle restrict
  int rn[m*o];
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < o; ++k) {
      rn[i + k*m] = xa[i][4][k];
    }
  }
  x.restrict(single(1), uint_vector{4}, result_span);
  x.restrict(ivec{1}, uint_vector{4}, result_iref);
  BOOST_CHECK_EQUAL(result_span.shape(), uint_vector({m, o}));
  BOOST_CHECK(std::equal(result_span.begin(), result_span.end(), rn));
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({m, o}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), rn));

  // Test boundary restrict
  int rom[n];
  for (std::size_t j = 0; j < n; ++j) {
    rom[j] = xa[5][j][7];
  }
  x.restrict(ivec{2, 0}, uint_vector{7, 5}, result_iref);
  BOOST_CHECK_EQUAL(result_iref.shape(), uint_vector({n}));
  BOOST_CHECK(std::equal(result_iref.begin(), result_iref.end(), rom));
}


BOOST_AUTO_TEST_CASE(test_restrict_join) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n][o];
  int ya[n][o];
  int_table x({m, n, o});
  int_table y({n, o});


  // Initialize the input arrays and tables
  int value = 2;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < o; ++k) {
        x({i,j,k}) = xa[i][j][k] = value++;
      }
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      y({j,k}) = ya[j][k] = value++;
    }
  }

  // Performs the restrict-sum operation using native arrays
  int rsum[n*o];
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      rsum[j + k*n] = xa[5][j][3] + ya[j][k];
    }
  }

  // Perform a generic restrict-join
  int_table result = y;
  std::plus<int> plus_op;
  x.restrict_join(plus_op, single(0), ivec{2, 0}, uint_vector{3, 5}, result);
  BOOST_CHECK_EQUAL(result.shape(), y.shape());
  BOOST_CHECK(std::equal(result.begin(), result.end(), rsum));
}
