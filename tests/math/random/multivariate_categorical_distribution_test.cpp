#define BOOST_TEST_MODULE multivariate_categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/factor/logarithmic_table.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>

#include <algorithm>
#include <numeric>

using namespace libgm;

using PTable = ProbabilityTable<double>;
using LTable = LogarithmicTable<double>;
using Dist = MultivariateCategoricalDistribution<double>;

std::size_t nsamples = 10000;
double tol = 0.02;

double marginal_diff(const Dist& d, const PTable& t) {
  std::mt19937 rng;
  PTable estimate(t.shape(), 0.0);
  for (std::size_t i = 0; i < nsamples; ++i) {
    std::vector<size_t> sample = d(rng);
    BOOST_REQUIRE_EQUAL(sample.size(), t.arity());
    ++estimate.param()(sample);
  }
  estimate /= double(nsamples);
  return std::inner_product(estimate.param().begin(), estimate.param().end(), t.param().begin(),
                            0.0, MaximumOp<double>(), AbsDifference<double>());
}

double conditional_diff(const Dist& d, const PTable& t) {
  std::mt19937 rng;
  PTable estimate(t.shape(), 0.0);
  for (std::size_t y = 0; y < t.shape().back(); ++y) {
    std::vector<size_t> tail(1, y);
    for (std::size_t i = 0; i < nsamples; ++i) {
      std::vector<size_t> sample = d(rng, tail);
      BOOST_REQUIRE_EQUAL(sample.size(), t.arity() - 1);
      sample.push_back(y);
      ++estimate.param()(sample);
    }
  }
  estimate /= double(nsamples);
  return std::inner_product(estimate.param().begin(), estimate.param().end(), t.param().begin(),
                            0.0, MaximumOp<double>(), AbsDifference<double>());
}

BOOST_AUTO_TEST_CASE(test_marginal) {
  PTable pt({2, 3}, {0.1, 0.05, 0.15, 0.25, 0.2, 0.25});
  LTable lt(pt.shape());
  std::transform(pt.param().begin(), pt.param().end(), lt.param().begin(), LogarithmOp<double>());
  BOOST_CHECK_SMALL(marginal_diff(Dist(pt), pt), tol);
  BOOST_CHECK_SMALL(marginal_diff(Dist(lt), pt), tol);
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  PTable pt({2, 3, 2},
            {0.1, 0.05, 0.15, 0.25, 0.2, 0.25,
             0.15, 0.25, 0.05, 0.2, 0.3, 0.05});
  LTable lt(pt.shape());
  std::transform(pt.param().begin(), pt.param().end(), lt.param().begin(), LogarithmOp<double>());
  BOOST_CHECK_SMALL(conditional_diff(Dist(pt), pt), tol);
  BOOST_CHECK_SMALL(conditional_diff(Dist(lt), pt), tol);
}
