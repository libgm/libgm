#define BOOST_TEST_MODULE table
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/table.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace libgm;

using IntTable = Table<int>;

BOOST_AUTO_TEST_CASE(test_accessors_and_copy) {
  const size_t n = 10;
  const size_t d = 3;

  IntTable x(Shape(n, d));
  BOOST_CHECK_EQUAL(x.size(), x.shape().product());

  int value = 0;
  for (const std::vector<size_t>& index : x.indices()) {
    x(index) = value++;
  }

  std::vector<int> seq(x.size());
  std::iota(seq.begin(), seq.end(), 0);
  BOOST_CHECK(std::equal(seq.begin(), seq.end(), x.begin()));

  IntTable y = x;
  BOOST_CHECK_EQUAL(x, y);
  y[0] = 20;
  BOOST_CHECK_NE(x, y);
}

BOOST_AUTO_TEST_CASE(test_fill_and_swap) {
  IntTable x({2, 3});
  IntTable y({3, 2});

  x.fill(7);
  y.fill(9);

  BOOST_CHECK_EQUAL(std::count(x.begin(), x.end(), 7), 6);
  BOOST_CHECK_EQUAL(std::count(y.begin(), y.end(), 9), 6);

  swap(x, y);
  BOOST_CHECK_EQUAL(x.shape(), Shape({3, 2}));
  BOOST_CHECK_EQUAL(y.shape(), Shape({2, 3}));
  BOOST_CHECK_EQUAL(std::count(x.begin(), x.end(), 9), 6);
  BOOST_CHECK_EQUAL(std::count(y.begin(), y.end(), 7), 6);
}

BOOST_AUTO_TEST_CASE(test_default_ctor_and_empty_indices) {
  IntTable x;
  BOOST_CHECK(x.empty());
  BOOST_CHECK_EQUAL(x.size(), 0);
  BOOST_CHECK_EQUAL(x.arity(), 0);
  BOOST_CHECK(x.data() == nullptr);
  BOOST_CHECK(x.begin() == x.end());

  auto range = x.indices();
  BOOST_CHECK(range.begin() == range.end());
}

BOOST_AUTO_TEST_CASE(test_constructors_and_size_dim) {
  IntTable a(Shape({2, 3}), 9);
  BOOST_CHECK_EQUAL(a.size(0), 2);
  BOOST_CHECK_EQUAL(a.size(1), 3);
  BOOST_CHECK_EQUAL(std::count(a.begin(), a.end(), 9), 6);

  IntTable b(Shape({2, 2}), {1, 2, 3, 4});
  BOOST_CHECK_EQUAL(b[0], 1);
  BOOST_CHECK_EQUAL(b[1], 2);
  BOOST_CHECK_EQUAL(b[2], 3);
  BOOST_CHECK_EQUAL(b[3], 4);

  std::vector<int> values = {5, 6, 7, 8, 9, 10};
  IntTable c(Shape({2, 3}), values.begin(), values.end());
  BOOST_CHECK(std::equal(values.begin(), values.end(), c.begin()));
}

BOOST_AUTO_TEST_CASE(test_reset_check_shape_and_index_api) {
  IntTable x({2, 3});
  std::iota(x.begin(), x.end(), 10);

  x.check_shape(Shape({2, 3}));
  BOOST_CHECK_THROW(x.check_shape(Shape({3, 2})), std::invalid_argument);

  // reset to same size keeps valid allocation and shape update
  int* old_ptr = x.data();
  x.reset(Shape({3, 2}));
  BOOST_CHECK_EQUAL(x.size(), 6);
  BOOST_CHECK_EQUAL(x.shape(), Shape({3, 2}));
  BOOST_CHECK(x.data() == old_ptr);

  // reset to different size reallocates
  x.reset(Shape({4, 2}));
  BOOST_CHECK_EQUAL(x.size(), 8);
  BOOST_CHECK_EQUAL(x.shape(), Shape({4, 2}));
  BOOST_CHECK(x.data() != old_ptr);

  // index calculation path via shape linear indexing
  x.fill(0);
  x({1, 1}) = 42;
  const size_t lin = x.shape().linear(std::vector<size_t>{1, 1});
  BOOST_CHECK_EQUAL(x[lin], 42);

  BOOST_CHECK_THROW((void)x({9, 0}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_move_and_self_assignment) {
  IntTable x({2, 2}, {1, 2, 3, 4});

  // copy self-assignment
  x = x;
  BOOST_CHECK_EQUAL(x.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(x[0], 1);
  BOOST_CHECK_EQUAL(x[3], 4);

  // move constructor
  IntTable y(std::move(x));
  BOOST_CHECK_EQUAL(y.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(y[0], 1);
  BOOST_CHECK_EQUAL(y[3], 4);

  // move assignment
  IntTable z;
  z = std::move(y);
  BOOST_CHECK_EQUAL(z.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(z[0], 1);
  BOOST_CHECK_EQUAL(z[3], 4);

  // move self-assignment
  z = std::move(z);
  BOOST_CHECK_EQUAL(z.shape(), Shape({2, 2}));
  BOOST_CHECK_EQUAL(z[0], 1);
  BOOST_CHECK_EQUAL(z[3], 4);
}
