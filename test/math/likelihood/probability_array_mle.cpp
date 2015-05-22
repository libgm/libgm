#define BOOST_TEST_MODULE probability_array_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_array_mle.hpp>

#include <libgm/math/likelihood/probability_array_ll.hpp>
#include <libgm/math/likelihood/range_ll.hpp>
#include <libgm/math/random/array_distribution.hpp>

#include <random>
#include <vector>

namespace libgm {
  template class probability_array_mle<double, 1>;
  template class probability_array_mle<double, 2>;
  template class probability_array_mle<float, 1>;
  template class probability_array_mle<float, 2>;
}

using namespace libgm;

std::size_t nsamples = 10000;
double tol = 0.01;

template <std::size_t N, typename Array>
double reconstruction_error(const Array& param) {
  // generate a few samples
  std::mt19937 rng;
  array_distribution<double, N> dist(param);
  typedef typename array_distribution<double, N>::result_type sample_type;
  std::vector<std::pair<sample_type, double>> samples;
  samples.reserve(nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  probability_array_mle<double, N> mle;
  Array estim = mle(samples, param.rows(), param.cols());

  typedef range_ll<probability_array_ll<double, N> > range_ll_type;
  double ll_truth = range_ll_type(param).value(samples);
  double ll_estim = range_ll_type(estim).value(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);

  return abs(estim-param).maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_mle1) {
  Eigen::Array<double, Eigen::Dynamic, 1> param(4);
  param << 0.1, 0.4, 0.3, 0.2;
  BOOST_CHECK_SMALL(reconstruction_error<1>(param), tol);
}

BOOST_AUTO_TEST_CASE(test_mle2) {
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> param(2, 3);
  param << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  BOOST_CHECK_SMALL(reconstruction_error<2>(param), tol);
}
