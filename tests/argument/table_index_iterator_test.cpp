#define BOOST_TEST_MODULE table_index_iterator
#include <boost/test/unit_test.hpp>

#include <libgm/argument/shape.hpp>
#include <libgm/iterator/table_index_iterator.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_iteration) {
  Shape shape = {2, 3};
  TableIndexIterator it(shape), end(shape.size());

  std::vector<size_t> expected0 = {0, 0};
  std::vector<size_t> expected1 = {1, 0};
  std::vector<size_t> expected2 = {0, 1};
  std::vector<size_t> expected3 = {1, 1};
  std::vector<size_t> expected4 = {0, 2};
  std::vector<size_t> expected5 = {1, 2};

  BOOST_CHECK(it != end);
  BOOST_CHECK((*it++) == expected0);
  BOOST_CHECK((*it++) == expected1);
  BOOST_CHECK((*it++) == expected2);
  BOOST_CHECK((*it++) == expected3);
  BOOST_CHECK((*it++) == expected4);
  BOOST_CHECK((*it++) == expected5);
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_empty_shape) {
  Shape empty_shape;
  TableIndexIterator it(empty_shape), end;

  BOOST_CHECK(it != end);
  BOOST_CHECK((*it).empty());
  BOOST_CHECK(++it == end);
}

BOOST_AUTO_TEST_CASE(test_digit_tracking_and_end_digit) {
  Shape shape = {2, 3};
  TableIndexIterator it(shape), end(shape.size());

  BOOST_CHECK_EQUAL(it.digit(), static_cast<size_t>(-1));

  ++it; // {1, 0}
  BOOST_CHECK_EQUAL(it.digit(), 0);
  BOOST_CHECK((*it == std::vector<size_t>{1, 0}));

  ++it; // {0, 1}
  BOOST_CHECK_EQUAL(it.digit(), 1);
  BOOST_CHECK((*it == std::vector<size_t>{0, 1}));

  ++it; // {1, 1}
  BOOST_CHECK_EQUAL(it.digit(), 0);
  ++it; // {0, 2}
  BOOST_CHECK_EQUAL(it.digit(), 1);
  ++it; // {1, 2}
  BOOST_CHECK_EQUAL(it.digit(), 0);
  ++it; // end
  BOOST_CHECK(it == end);
  BOOST_CHECK_EQUAL(it.digit(), shape.size());
}

BOOST_AUTO_TEST_CASE(test_postfix_and_arrow_and_copy_assign) {
  Shape shape = {2, 2};
  TableIndexIterator it(shape), end(shape.size());

  BOOST_CHECK_EQUAL(it->size(), 2);
  BOOST_CHECK((*it == std::vector<size_t>{0, 0}));

  TableIndexIterator before = it++;
  BOOST_CHECK((*before == std::vector<size_t>{0, 0}));
  BOOST_CHECK((*it == std::vector<size_t>{1, 0}));

  TableIndexIterator copy = it;
  BOOST_CHECK(copy == it);
  ++copy;
  BOOST_CHECK(copy != it);

  copy = it;
  BOOST_CHECK(copy == it);

  // Exhaust iterator to end to verify normal termination in this shape.
  ++it; // {0, 1}
  ++it; // {1, 1}
  ++it; // end
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_three_dimensional_carry_sequence) {
  Shape shape = {2, 2, 2};
  TableIndexIterator it(shape), end(shape.size());

  std::vector<std::vector<size_t>> expected = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
  };

  for (const auto& idx : expected) {
    BOOST_CHECK(it != end);
    BOOST_CHECK(*it == idx);
    ++it;
  }
  BOOST_CHECK(it == end);
}
