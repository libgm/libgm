#define BOOST_TEST_MODULE categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/bivariate_categorical_distribution.hpp>

namespace libgm {
  template class bivariate_categorical_distribution<double>;
  template class bivariate_categorical_distribution<float>;
}

using namespace libgm;

typedef bivariate_categorical_distribution<> dist_type;

std::size_t nsamples = 10000;
double tol = 0.01;

double marginal_diff(const dist_type& d, const dense_matrix<>& a) {
  std::mt19937 rng;
  dense_matrix<> estimate = dense_matrix<>::Zero(a.rows(), a.cols());
  for (std::size_t i = 0; i < nsamples; ++i) {
    std::pair<std::size_t, std::size_t> sample = d(rng);
    ++estimate(sample.first, sample.second);
  }
  estimate /= double(nsamples);
  return (estimate - a).array().abs().maxCoeff();
}

double conditional_diff(const dist_type& d, const dense_matrix<>& a) {
  std::mt19937 rng;
  dense_matrix<> estimate = dense_matrix<>::Zero(a.rows(), a.cols());
  for (std::ptrdiff_t tail = 0; tail < a.cols(); ++tail) {
    for (std::size_t i = 0; i < nsamples; ++i) {
      ++estimate(d(rng, tail), tail);
    }
  }
  estimate /= double(nsamples);
  return (estimate - a).array().abs().maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_marginal) {
  dense_matrix<> a(2, 3);
  a << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  BOOST_CHECK_SMALL(marginal_diff(dist_type(a, prob_tag()), a), tol);
  BOOST_CHECK_SMALL(marginal_diff(dist_type(a.array().log(), log_tag()), a), tol);
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  dense_matrix<> a(3, 2);
  a << 0.1, 0.25, 0.4, 0.60, 0.5, 0.15;
  BOOST_CHECK_SMALL(conditional_diff(dist_type(a, prob_tag()), a), tol);
  BOOST_CHECK_SMALL(conditional_diff(dist_type(a.array().log(), log_tag()), a), tol);
}
