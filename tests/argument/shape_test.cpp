#define BOOST_TEST_MODULE shape
#include <boost/test/unit_test.hpp>

#include <libgm/argument/shape.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_prefix_suffix_and_sums) {
  Shape s = {2, 3, 4, 5};

  BOOST_CHECK(s.has_prefix(Shape({2, 3})));
  BOOST_CHECK(!s.has_prefix(Shape({2, 4})));
  BOOST_CHECK(s.has_suffix(Shape({4, 5})));
  BOOST_CHECK(!s.has_suffix(Shape({3, 5})));

  BOOST_CHECK(s.prefix(2) == Shape({2, 3}));
  BOOST_CHECK(s.suffix(2) == Shape({4, 5}));
  BOOST_CHECK_EQUAL(s.prefix_sum(3), 9);
  BOOST_CHECK_EQUAL(s.suffix_sum(2), 9);
  BOOST_CHECK_EQUAL(s.sum(), 14);
  BOOST_CHECK_EQUAL(s.product(), 120);

  BOOST_CHECK_THROW(s.prefix(5), std::invalid_argument);
  BOOST_CHECK_THROW(s.suffix(5), std::invalid_argument);
  BOOST_CHECK_THROW(s.prefix_sum(5), std::invalid_argument);
  BOOST_CHECK_THROW(s.suffix_sum(5), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_prefix_suffix_size) {
  Shape s = {2, 3, 4, 5};

  BOOST_CHECK_EQUAL(s.prefix_size(0), 0u);
  BOOST_CHECK_EQUAL(s.prefix_size(2), 1u);
  BOOST_CHECK_EQUAL(s.prefix_size(5), 2u);
  BOOST_CHECK_EQUAL(s.prefix_size(14), 4u);

  BOOST_CHECK_EQUAL(s.suffix_size(0), 0u);
  BOOST_CHECK_EQUAL(s.suffix_size(5), 1u);
  BOOST_CHECK_EQUAL(s.suffix_size(9), 2u);
  BOOST_CHECK_EQUAL(s.suffix_size(14), 4u);

  BOOST_CHECK_THROW(s.prefix_size(1), std::invalid_argument);
  BOOST_CHECK_THROW(s.prefix_size(6), std::invalid_argument);
  BOOST_CHECK_THROW(s.prefix_size(15), std::invalid_argument);
  BOOST_CHECK_THROW(s.suffix_size(1), std::invalid_argument);
  BOOST_CHECK_THROW(s.suffix_size(6), std::invalid_argument);
  BOOST_CHECK_THROW(s.suffix_size(15), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_index_and_linear) {
  Shape s = {2, 3, 4};
  std::vector<size_t> idx = {1, 2, 3};

  size_t linear = s.linear(idx);
  BOOST_CHECK_EQUAL(linear, 23);
  BOOST_CHECK(s.index(linear) == idx);
  BOOST_CHECK_EQUAL(s.linear_front({1, 2}), 5);
  BOOST_CHECK_EQUAL(s.linear_back({2, 3}), 22);

  Dims dims = make_dims({0, 2});
  BOOST_CHECK(s.has_select(dims, Shape({2, 4})));
  BOOST_CHECK_EQUAL(s.linear(dims, {1, 3}), 19);

  BOOST_CHECK_THROW(s.linear({1, 2}), std::invalid_argument);
  BOOST_CHECK_THROW(s.linear_front({1, 2, 0, 1}), std::invalid_argument);
  BOOST_CHECK_THROW(s.linear_back({0, 0, 0, 0}), std::invalid_argument);
  BOOST_CHECK_THROW(s.index(24), std::invalid_argument);
  BOOST_CHECK_THROW(s.linear(dims, {2, 0}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_select_omit_spans_and_join) {
  Shape s = {2, 3, 4, 5};
  Dims sel = make_dims({0, 2});
  Dims omit = make_dims({1, 3});

  BOOST_CHECK(s.select(sel) == Shape({2, 4}));
  BOOST_CHECK(s.omit(omit) == Shape({2, 4}));

  Dims spans_dims = make_dims({0, 1, 3});
  Spans spans = s.spans(spans_dims);
  BOOST_CHECK_EQUAL(spans.size(), 2);
  BOOST_CHECK_EQUAL(spans[0].start, 0);
  BOOST_CHECK_EQUAL(spans[0].length, 5);
  BOOST_CHECK_EQUAL(spans[1].start, 9);
  BOOST_CHECK_EQUAL(spans[1].length, 5);
  BOOST_CHECK_EQUAL(spans.sum(), 10);

  Shape a = {2, 3};
  Shape b = {3, 5};
  Dims i = make_dims({0, 1});
  Dims j = make_dims({1, 2});
  BOOST_CHECK(join(a, b, i, j) == Shape({2, 3, 5}));
  BOOST_CHECK((a + b) == Shape({2, 3, 3, 5}));

  BOOST_CHECK_THROW(join(a, Shape({4, 5}), i, j), std::invalid_argument);
  BOOST_CHECK_THROW(s.select(make_dims({0, 10})), std::invalid_argument);
  BOOST_CHECK(s.omit(make_dims({0, 1, 2, 3, 10})) == Shape());
  BOOST_CHECK_THROW(s.spans(make_dims({0, 10})), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_spans_use_flat_offsets_for_non_unit_shape) {
  Shape s = {2, 3, 4, 5};

  // Selected dims: 1 and 3 (non-contiguous, non-unit sizes).
  // Flat offsets should be:
  // dim 1 starts at 2, length 3
  // dim 3 starts at 2 + 3 + 4 = 9, length 5
  Spans spans = s.spans(make_dims({1, 3}));
  BOOST_CHECK_EQUAL(spans.size(), 2);
  BOOST_CHECK_EQUAL(spans[0].start, 2);
  BOOST_CHECK_EQUAL(spans[0].length, 3);
  BOOST_CHECK_EQUAL(spans[1].start, 9);
  BOOST_CHECK_EQUAL(spans[1].length, 5);
}

BOOST_AUTO_TEST_CASE(test_stream_and_product_overflow) {
  Shape s = {2, 3, 4};
  std::ostringstream out;
  out << s;
  BOOST_CHECK_EQUAL(out.str(), "Shape([2, 3, 4])");

  Shape huge = {std::numeric_limits<size_t>::max(), 2};
  BOOST_CHECK_THROW(huge.product(), std::out_of_range);
}
