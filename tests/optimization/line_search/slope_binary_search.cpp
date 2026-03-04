#define BOOST_TEST_MODULE slope_binary_search
#include <boost/test/unit_test.hpp>

#include <libgm/optimization/line_search/slope_binary_search.hpp>

#include "../quadratic_objective.hpp"

#include <random>

namespace libgm {
  template class slope_binary_search<vec_type>;
}

using namespace libgm;
typedef line_search_result<double> result_type;

// test the high-accuracy search
BOOST_AUTO_TEST_CASE(test_slope_binary_search) {
  quadratic_objective objective(vec2(5, 4), mat22(1, 0, 0, 1));
  slope_binary_search<vec_type> search;
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

// test that we reach Wolfe conditions
// by shooting from random points and verifying the conditions manually
BOOST_AUTO_TEST_CASE(test_wolfe) {
  quadratic_objective objective(vec2(-1, 1), mat22(2, 1, 1, 2));
  slope_binary_search<vec_type> search(bracketing_parameters<double>(),
                                       wolfe<double>::conjugate_gradient());
  search.objective(&objective);

  std::size_t nlines = 20;
  std::mt19937 rng;
  std::uniform_real_distribution<> unif(-5, 5);
  for (std::size_t i = 0; i < nlines; ++i) {
    vec_type src = vec2(unif(rng), unif(rng));
    vec_type dir = vec2(unif(rng), unif(rng));
    if (objective.gradient(src).dot(dir) > 0) {
      dir = -dir;
    }
    double f0 = objective.value(src);
    double g0 = objective.gradient(src).dot(dir);
    result_type r = search.step(src, dir, objective.init(src, dir));
    double fa = objective.value(src+r.step*dir);
    double ga = objective.gradient(src+r.step*dir).dot(dir);

    // verify the accuracy of results and validity of the Wolfe conditions
    BOOST_CHECK_LT(r.value, objective.value(src));
    BOOST_CHECK_CLOSE(r.value, fa, 1e-2);
    BOOST_CHECK_LE(fa, f0 + search.wolfe().c1 * r.step * g0);
    BOOST_CHECK_LE(std::abs(ga), search.wolfe().c2 * std::abs(g0));
  }
}
