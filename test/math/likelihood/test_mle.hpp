#ifndef LIBGM_TEST_MLE_HPP
#define LIBGM_TEST_MLE_HPP

#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/math/likelihood/range_ll.hpp>

#include <random>
#include <vector>

using namespace libgm;

template <typename Distribution, typename MLE, typename LL, typename Shape>
double reconstruction_error(std::size_t nsamples,
                            const typename Distribution::param_type& param,
                            Shape shape) {
  // generate a few samples
  std::mt19937 rng;
  Distribution dist(param, prob_tag());
  std::vector<std::pair<typename Distribution::result_type, double>> samples;
  samples.reserve(nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  typename Distribution::param_type estim = MLE()(samples, shape);

  typedef range_ll<LL> range_ll_type;
  double ll_truth = range_ll_type(param).value(samples);
  double ll_estim = range_ll_type(estim).value(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);

  return
    std::inner_product(estim.data(), estim.data() + estim.size(),
                       param.data(),
                       0.0, maximum<double>(), abs_difference<double>());
}

#endif
