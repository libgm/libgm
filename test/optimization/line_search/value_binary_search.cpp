#define BOOST_TEST_MODULE value_binary_search
#include <boost/test/unit_test.hpp>

#include <libgm/optimization/line_search/value_binary_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

namespace libgm {
  template class value_binary_search<vec_type>;
}

using namespace libgm;
typedef line_search_result<double> result_type;

BOOST_AUTO_TEST_CASE(test_value_binary_search) {
  quadratic_objective objective(vec2(5, 4), mat22(1, 0, 0, 1));
  value_binary_search<vec_type> search;
  search.objective(&objective);
  result_type horiz = search.step(vec2(3.987, 3), vec2(1, 0),
                                  objective.init(vec2(3.987, 3), vec2(1, 0)));
  BOOST_CHECK_CLOSE(horiz.step, 1.013, 1e-3);
  BOOST_CHECK_CLOSE(horiz.value, 0.5, 1e-3);
  result_type diag = search.step(vec2(1, 0), vec2(1, 1),
                                 objective.init(vec2(1, 0), vec2(1, 1)));
  BOOST_CHECK_CLOSE(diag.step, 4.0, 1e-3);
  BOOST_CHECK_SMALL(diag.value, 1e-5);
}
