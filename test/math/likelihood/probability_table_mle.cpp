#define BOOST_TEST_MODULE probability_table_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_table_mle.hpp>

#include <libgm/math/likelihood/probability_table_ll.hpp>
#include <libgm/math/likelihood/range_ll.hpp>
#include <libgm/math/random/table_distribution.hpp>

#include <random>
#include <vector>

namespace libgm {
  template class probability_table_mle<double>;
  template class probability_table_mle<float>;
}

using namespace libgm;

std::size_t nsamples = 10000;
double tol = 0.01;

BOOST_AUTO_TEST_CASE(test_mle) {
  table<double> param({2, 3}, {0.1, 0.05, 0.15, 0.25, 0.2, 0.25});

  // generate a few samples
  std::mt19937 rng;
  table_distribution<> dist(param);
  std::vector<std::pair<finite_index, double>> samples;
  samples.reserve(nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  table<double> estim = probability_table_mle<>()(samples, param.shape());
  double diff =
    std::inner_product(estim.begin(), estim.end(), param.begin(),
                       0.0, maximum<double>(), abs_difference<double>());
  BOOST_CHECK_SMALL(diff, tol);

  typedef range_ll<probability_table_ll<> > range_ll_type;
  double ll_truth = range_ll_type(param).value(samples);
  double ll_estim = range_ll_type(estim).value(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);
}
