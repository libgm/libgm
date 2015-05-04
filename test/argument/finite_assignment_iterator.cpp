#define BOOST_TEST_MODULE finite_assignment_iterator
#include <boost/test/unit_test.hpp>

#include <libgm/argument/finite_assignment_iterator.hpp>
#include <libgm/argument/universe.hpp>

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_assignment<>);
BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_assignment_iterator<>);

BOOST_AUTO_TEST_CASE(test_iteration) {
  universe u;
  variable v = u.new_finite_variable("v", 2);
  variable x = u.new_finite_variable("x", 3);
  variable y = u.new_vector_variable("y", 2);

  // Test the assignment iterator
  finite_assignment_iterator<> it({v, x}), end;
  finite_assignment<> fa;
  for (size_t i = 0; i < 3; ++i) {
    fa[x] = i;
    for (size_t j = 0; j < 2; ++j) {
      fa[v] = j;
      BOOST_CHECK_NE(it, end);
      BOOST_CHECK_EQUAL(*it++, fa);
    }
  }
  BOOST_CHECK_EQUAL(it, end);

  // Test the empty iterator
  it = finite_assignment_iterator<>(domain());
  fa.clear();
  BOOST_CHECK_EQUAL(*it, fa);
  BOOST_CHECK_EQUAL(++it, end);
}
