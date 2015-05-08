#define BOOST_TEST_MODULE softmax_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/softmax_ll.hpp>
#include <libgm/math/likelihood/softmax_mle.hpp>
#include <libgm/math/random/softmax_distribution.hpp>

#include <random>

namespace libgm {
  template class softmax_mle<double>;
  template class softmax_mle<float>;
  template class softmax_ll<double>;
  template class softmax_ll<float>;
}

using namespace libgm;

std::size_t nsamples = 1000;
double tol = 0.1;

BOOST_AUTO_TEST_CASE(test_mle) {
  softmax_param<double> param(3, 4);
  param.bias() << 1.0, 0.5, 1.5;
  param.weight() << 3.0, 2.0, 1.0, -0.5,
                    2.0, 1.5, 1.0, -1.0,
                    0.1, 0.5, 1.5, -2.0;

  // generate a few samples
  std::mt19937 rng;
  std::uniform_real_distribution<double> unif;
  softmax_distribution<double> dist(param);
  typedef hybrid_index<double> vec_type;
  std::vector<std::pair<vec_type, double>> samples;
  samples.reserve(nsamples);
  vec_type sample(1, 4);
  for (std::size_t i = 0; i < nsamples; ++i) {
    for (std::size_t j = 0; j < 4; ++j) { sample.vector()[j] = unif(rng); }
    sample.finite()[0] = dist(rng, sample.vector());
    samples.emplace_back(sample, 1.0);
  }

  // compute the MLE and evaluate its log-likelihood on the training set
  softmax_mle<double> mle(0.01, 100, true);
  softmax_param<double> estim(3, 4);
  mle.estimate(samples, estim);

  std::cout << "Solution: " << estim << std::endl;

  typedef range_ll<softmax_ll<double> > range_ll_type;
  double ll_truth = range_ll_type(param).value(samples);
  double ll_estim = range_ll_type(estim).value(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);
}
