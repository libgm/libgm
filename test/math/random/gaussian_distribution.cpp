#define BOOST_TEST_MODULE gaussian_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/gaussian_distribution.hpp>

namespace libgm {
  template class gaussian_distribution<double>;
  template class gaussian_distribution<float>;
}

using namespace libgm;

typedef moment_gaussian_param<double> param_type;
typedef gaussian_distribution<double> distribution_type;

size_t nsamples = 10000;
double tol = 0.05;

BOOST_AUTO_TEST_CASE(test_marginal) {
  param_type param(3, 0);
  param.mean << 1.0, 3.0, 2.0;
  param.cov  << 3.0, 2.0, 1.0,
                2.0, 1.5, 1.0,
                1.0, 1.0, 1.5;

  distribution_type d(param);
  dynamic_vector<double> mean; mean.setZero(3);
  dynamic_matrix<double> cov; cov.setZero(3, 3);
  std::mt19937 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    dynamic_vector<double> sample = d(rng);
    mean += sample;
    cov += sample * sample.transpose();
  }
  mean /= nsamples;
  cov /= nsamples;
  cov -= mean * mean.transpose();
  BOOST_CHECK_SMALL((param.mean - mean).cwiseAbs().maxCoeff(), tol);
  BOOST_CHECK_SMALL((param.cov - cov).cwiseAbs().maxCoeff(), tol);
}
