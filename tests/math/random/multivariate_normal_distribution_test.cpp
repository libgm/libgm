#define BOOST_TEST_MODULE multivariate_normal_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/multivariate_normal_distribution.hpp>
#include <libgm/factor/moment_gaussian.hpp>

using namespace libgm;

using MGaussian = MomentGaussian<double>;
using Dist = MultivariateNormalDistribution<double>;
using Vec = Vector<double>;
using Mat = Matrix<double>;

size_t nsamples = 10000;
double tol = 0.05;

BOOST_AUTO_TEST_CASE(test_marginal) {
  Vec mean0{{1.0, 3.0, 2.0}};
  Mat cov0(3, 3);
  cov0 << 3.0, 2.0, 1.0,
          2.0, 1.5, 1.0,
          1.0, 1.0, 1.5;

  MGaussian mg(Shape{3}, mean0, cov0);
  Dist d(mg);
  Vec mean = Vec::Zero(3);
  Mat cov = Mat::Zero(3, 3);
  std::mt19937 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    Vec sample = d(rng);
    mean += sample;
    cov += sample * sample.transpose();
  }
  mean /= double(nsamples);
  cov /= double(nsamples);
  cov -= mean * mean.transpose();
  BOOST_CHECK_SMALL((mean0 - mean).cwiseAbs().maxCoeff(), tol);
  BOOST_CHECK_SMALL((cov0 - cov).cwiseAbs().maxCoeff(), tol);
}
