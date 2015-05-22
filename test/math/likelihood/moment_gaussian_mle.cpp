#define BOOST_TEST_MODULE moment_gaussian_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/moment_gaussian_mle.hpp>

#include <libgm/math/likelihood/moment_gaussian_ll.hpp>
#include <libgm/math/likelihood/range_ll.hpp>
#include <libgm/math/random/gaussian_distribution.hpp>

#include <random>
#include <vector>

namespace libgm {
  template class moment_gaussian_mle<double>;
  template class moment_gaussian_mle<float>;
}

using namespace libgm;

std::size_t nsamples = 10000;
double tol = 0.05;

BOOST_AUTO_TEST_CASE(test_mle) {
  typedef dynamic_vector<double> vec_type;
  moment_gaussian_param<> param(3, 0);
  param.mean << 1.0, 3.0, 2.0;
  param.cov  << 3.0, 2.0, 1.0,
                2.0, 1.5, 1.0,
                1.0, 1.0, 1.5;

  // generate a few samples
  std::mt19937 rng;
  gaussian_distribution<> dist(param);
  std::vector<std::pair<vec_type, double>> samples;
  samples.reserve(nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  moment_gaussian_param<> estim = moment_gaussian_mle<>()(samples, 3);
  BOOST_CHECK_SMALL((param.mean - estim.mean).cwiseAbs().maxCoeff(), tol);
  BOOST_CHECK_SMALL((param.cov - estim.cov).cwiseAbs().maxCoeff(), tol);

  // check if the log-likelihoods are close
  typedef range_ll<moment_gaussian_ll<> > range_ll_type;
  double ll_truth = range_ll_type(param).value(samples);
  double ll_estim = range_ll_type(estim).value(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);
}
