#define BOOST_TEST_MODULE backtracking_line_search
#include <boost/test/unit_test.hpp>

#include <libgm/optimization/line_search/backtracking_line_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

namespace libgm {
  template class backtracking_line_search<vec_type>;
}

using namespace libgm;
typedef line_search_result<double> result_type;

BOOST_AUTO_TEST_CASE(test_exponential_decay_search) {
  quadratic_objective objective(vec2(5, 4), mat22(1, 0, 0, 1));
  backtracking_line_search_parameters<double> params(0.3, 0.5);
  backtracking_line_search<vec_type> search(params);
  search.objective(&objective);
  
  result_type r = search.step(vec2(1, 2), vec2(1, 0.5),
                              objective.init(vec2(1, 2), vec2(1, 0.5)));
  BOOST_CHECK_CLOSE(r.step, 1.0, 1e-6);
  BOOST_CHECK_CLOSE(r.value, 5.625, 1e-6);
}
