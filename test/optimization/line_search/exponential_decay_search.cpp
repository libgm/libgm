#define BOOST_TEST_MODULE exponential_decay_search
#include <boost/test/unit_test.hpp>

#include <libgm/optimization/line_search/exponential_decay_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

namespace libgm {
  template class exponential_decay_search<vec_type>;
}

using namespace libgm;
typedef line_search_result<double> result_type;

BOOST_AUTO_TEST_CASE(test_exponential_decay_search) {
  quadratic_objective objective(vec2(5, 4), mat22(1, 0, 0, 1));
  exponential_decay_search_parameters<double> params(0.5, 0.1);
  exponential_decay_search<vec_type> search(params);
  search.objective(&objective);
  
  result_type r1 = search.step(vec2(1, 2), vec2(1, 0), result_type());
  BOOST_CHECK_CLOSE(r1.step, 0.5, 1e-6);
  BOOST_CHECK_CLOSE(r1.value, objective.value(vec2(1.5, 2)), 1e-6);
  
  result_type r2 = search.step(vec2(4, 3), vec2(1, 1), result_type());
  BOOST_CHECK_CLOSE(r2.step, 0.05, 1e-6);
  BOOST_CHECK_CLOSE(r2.value, objective.value(vec2(4.05, 3.05)), 1e-6);
}
