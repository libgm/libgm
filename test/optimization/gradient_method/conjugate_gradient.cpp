#define BOOST_TEST_MODULE conjugate_gradient
#include <boost/test/unit_test.hpp>

#include <libgm/optimization/gradient_method/conjugate_gradient.hpp>
#include <libgm/optimization/line_search/slope_binary_search.hpp>

#include "../quadratic_objective.hpp"

namespace libgm {
  template class conjugate_gradient<vec_type>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_standard) {
  quadratic_objective objective(vec2(2, 3), mat22(2, 1, 1, 2));
  line_search<vec_type>* search = new slope_binary_search<vec_type>;
  conjugate_gradient<vec_type> cg(search);
  cg.objective(&objective);
  cg.solution(vec2(0, 0));
  for (size_t it = 0; it < 5 && !cg.converged(); ++it) {
    line_search_result<double> result = cg.iterate();
    std::cout << "Iteration " << it << ", result " << result << std::endl;
  }
  std::cout << "Standard: " << cg.solution().transpose() << std::endl;
  BOOST_CHECK(cg.converged());
  BOOST_CHECK_SMALL((cg.solution() - objective.ctr).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(test_preconditioned) {
  quadratic_objective objective(vec2(2, 3), mat22(2, 1, 1, 2));
  line_search<vec_type>* search = new slope_binary_search<vec_type>;
  conjugate_gradient<vec_type> cg(search, {1e-6, true});
  cg.objective(&objective);
  cg.solution(vec2(0, 0));
  for (size_t it = 0; it < 5 && !cg.converged(); ++it) {
    line_search_result<double> result = cg.iterate();
    std::cout << "Iteration " << it << ", result " << result << std::endl;
  }
  std::cout << "Estimate: " << cg.solution().transpose() << std::endl;
  BOOST_CHECK(cg.converged());
  BOOST_CHECK_SMALL((cg.solution() - objective.ctr).norm(), 1e-6);
}
