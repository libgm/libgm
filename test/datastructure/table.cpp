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


BOOST_AUTO_TEST_CASE(test_accessors) {
  const std::size_t n = 10;
  const std::size_t d = 3;
  const std::size_t m = pow(d, n);

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
  int_table z({2});

  // Fill
  x.fill(3);
  BOOST_CHECK_EQUAL(std::count(x.begin(), x.end(), 3), 4);

  // Unary transform
  std::iota(x.begin(), x.end(), 2);
  x.transform(libgm::incremented_by<int>(3));
  BOOST_CHECK_EQUAL(x[0], 5);
  BOOST_CHECK_EQUAL(x[1], 6);
  BOOST_CHECK_EQUAL(x[2], 7);
  BOOST_CHECK_EQUAL(x[3], 8);

  // Binary transform
  std::iota(x.begin(), x.end(), 1);
  std::iota(y.begin(), y.end(), 3);
  x.transform(y, std::plus<int>());
  BOOST_CHECK_EQUAL(x[0], 4);
  BOOST_CHECK_EQUAL(x[1], 6);
  BOOST_CHECK_EQUAL(x[2], 8);
  BOOST_CHECK_EQUAL(x[3], 10);

  // Accumulate
  BOOST_CHECK_EQUAL(x.accumulate(1, std::plus<int>()), 29);

  // Transform-accumulate
  std::iota(x.begin(), x.end(), 2);
  std::size_t sum = x.transform_accumulate(0,
                                      libgm::incremented_by<int>(3),
                                      std::plus<int>());
  BOOST_CHECK_EQUAL(sum, 26);

  // Restrict
  z.restrict(x, 2);
  BOOST_CHECK_EQUAL(z[0], 4);
  BOOST_CHECK_EQUAL(z[1], 5);
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

  /*
  param_type cond = p.condition({1});
  BOOST_CHECK_EQUAL(cond.arity(), 1);
  BOOST_CHECK_EQUAL(cond.size(), 1);
  BOOST_CHECK_EQUAL(cond[0], 2);
  */
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

  uint_vector x_map = {0, 1};
  uint_vector y_map = {1, 2};
  uint_vector z_map = {2, 0};
  std::plus<int> op;

  // Compute the sums with nested loops
  int_table nested({m, n, o});
  table_join<int,int,std::plus<int> >(nested, x, y, x_map, y_map, op)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), sum_xy));
  table_join_inplace<int,int,std::plus<int> >(nested, z, z_map, op)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), sum_xyz));

  // Compute the sums with flat loop
  int_table flat({m, n, o});
  table_join<int,int,std::plus<int> >(flat, x, y, x_map, y_map, op).loop();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), sum_xy));
  table_join_inplace<int,int,std::plus<int> >(flat, z, z_map, op).loop();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), sum_xyz));
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

  // Arguments of the operation
  uint_vector dim_map = {2, 0};
  std::plus<int> op;

  // Performs the aggregate with nested loops
  int_table nested({o, m});
  nested.fill(0);
  table_aggregate<int,int,std::plus<int> >(nested, x, dim_map, op)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), sum));

  // Performs the aggregate with flat loop
  int_table flat({o, m});
  flat.fill(0);
  table_aggregate<int,int,std::plus<int> >(flat, x, dim_map, op).loop();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), sum));
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

  uint_vector x_map = {0, 1};
  uint_vector y_map = {1, 2};
  uint_vector r_map = {2, 0};
  uint_vector z_shape = {m, n, o};
  std::multiplies<int> join_op;
  std::plus<int> agg_op;

  // Compute the aggregates with nested loops
  int_table nested({o, m});
  nested.fill(0);
  table_join_aggregate<int,std::multiplies<int>,std::plus<int> >
    (nested, x, y, r_map, x_map, y_map, z_shape, join_op, agg_op)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), sum));

  // Compute the sums with flat loop
  int_table flat({o, m});
  flat.fill(0);
  table_join_aggregate<int,std::multiplies<int>,std::plus<int> >
    (flat, x, y, r_map, x_map, y_map, z_shape, join_op, agg_op).loop();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), sum));
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

  // Performs the restrict operation using native arrays
  int result[o*n];
  for (std::size_t k = 0; k < o; ++k) {
    for (std::size_t j = 0; j < n; ++j) {
      result[k + j*o] = xa[2][j][k];
    }
  }

  // Dimension mapping
  uint_vector x_map = {std::size_t(-1), 1, 0};

  // Performs the restrict operation using nested loops
  int_table nested({o, n});
  table_restrict<int>(nested, x, x_map, 2)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), result));

  // Performs the restrict operation using a flat loop
  int_table flat({o, n});
  table_restrict<int>(flat, x, x_map, 2).loop();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), result));
}


BOOST_AUTO_TEST_CASE(test_restrict_join) {
  const std::size_t m = 10;
  const std::size_t n = 8;
  const std::size_t o = 9;

  // Input arrays and tables
  int xa[m][n];
  int ya[n][o];
  int_table x({m, n});
  int_table y({n, o});


  // Initialize the input arrays and tables
  int value = 2;
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

  // Performs the restrict-sum operation using native arrays
  int result[n*o];
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < o; ++k) {
      result[j + k*n] = xa[2][j] + ya[j][k];
    }
  }

  // Dimension mapping
  uint_vector x_map = {std::size_t(-1), 0};
  std::plus<int> op;

  // Performs the restrict-join operation using nested loops
  int_table nested = y;
  table_restrict_join<int,int,std::plus<int> >(nested, x, x_map, 2, op)();
  BOOST_CHECK(std::equal(nested.begin(), nested.end(), result));

  // Performs the restrict-join operation using flat loop
  int_table flat = y;
  table_restrict_join<int,int,std::plus<int> >(flat, x, x_map, 2, op)();
  BOOST_CHECK(std::equal(flat.begin(), flat.end(), result));
}
