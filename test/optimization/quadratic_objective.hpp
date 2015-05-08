#ifndef LIBGM_TEST_QUADRATIC_OBJECTIVE_HPP
#define LIBGM_TEST_QUADRATIC_OBJECTIVE_HPP

#include <libgm/optimization/gradient_objective/gradient_objective.hpp>
#include <libgm/optimization/line_search/line_search_result.hpp>

#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/optimization.hpp>

#include "../math/eigen/helpers.hpp"

typedef libgm::dynamic_matrix<double> mat_type;
typedef libgm::dynamic_vector<double> vec_type;

// a quadratic objective 0.5 * (x-ctr)^T cov (x-ctr)
struct quadratic_objective
  : public libgm::gradient_objective<vec_type> {

  vec_type ctr;
  mat_type cov;

  quadratic_objective(const vec_type& ctr, const mat_type& cov)
    : ctr(ctr), cov(cov) { }

  double value(const vec_type& x) override {
    vec_type diff = x - ctr;
    return 0.5 * diff.dot(cov * diff);
  }

  libgm::real_pair<> value_slope(const vec_type& x, const vec_type& dir) override {
    vec_type diff = x - ctr;
    double value = 0.5 * diff.dot(cov * diff);
    double slope = dir.dot(cov * diff);
    return { value, slope };
  }

  void add_gradient(const vec_type& x, vec_type& g) override {
    g += cov * (x - ctr);
  }

  void add_hessian_diag(const vec_type& x, vec_type& h) override {
    h += cov.diagonal();
  }

  libgm::gradient_objective_calls calls() const override {
    return libgm::gradient_objective_calls();
  }

  libgm::line_search_result<double> init(const vec_type& x, const vec_type& dir) {
    libgm::line_search_result<double> result(0);
    std::tie(result.value, result.slope) = value_slope(x, dir);
    return result;
  }


}; // struct quadratic_objective

#endif
