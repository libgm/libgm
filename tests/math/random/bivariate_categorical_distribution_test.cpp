#define BOOST_TEST_MODULE categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/bivariate_categorical_distribution.hpp>
#include <libgm/factor/logarithmic_matrix.hpp>
#include <libgm/factor/probability_matrix.hpp>

using namespace libgm;

using Dist = BivariateCategoricalDistribution<double>;
using Mat = Matrix<double>;
using PMat = ProbabilityMatrix<double>;
using LMat = LogarithmicMatrix<double>;

std::size_t nsamples = 10000;
double tol = 0.01;

double marginal_diff(const Dist& d, const Mat& a) {
  std::mt19937 rng;
  Mat estimate = Mat::Zero(a.rows(), a.cols());
  for (std::size_t i = 0; i < nsamples; ++i) {
    std::pair<std::size_t, std::size_t> sample = d(rng);
    ++estimate(sample.first, sample.second);
  }
  estimate /= double(nsamples);
  return (estimate - a).array().abs().maxCoeff();
}

double conditional_diff(const Dist& d, const Mat& a) {
  std::mt19937 rng;
  Mat estimate = Mat::Zero(a.rows(), a.cols());
  for (std::ptrdiff_t tail = 0; tail < a.cols(); ++tail) {
    for (std::size_t i = 0; i < nsamples; ++i) {
      ++estimate(d(rng, tail), tail);
    }
  }
  estimate /= double(nsamples);
  return (estimate - a).array().abs().maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_marginal) {
  Mat a(2, 3);
  a << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  BOOST_CHECK_SMALL(marginal_diff(Dist(PMat(a)), a), tol);
  BOOST_CHECK_SMALL(marginal_diff(Dist(LMat(a.array().log())), a), tol);
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  Mat a(3, 2);
  a << 0.1, 0.25, 0.4, 0.60, 0.5, 0.15;
  BOOST_CHECK_SMALL(conditional_diff(Dist(PMat(a)), a), tol);
  BOOST_CHECK_SMALL(conditional_diff(Dist(LMat(a.array().log())), a), tol);
}
