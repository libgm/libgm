#define BOOST_TEST_MODULE multivariate_categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/multivariate_categorical_distribution.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>

#include <algorithm>
#include <numeric>

namespace libgm {
  template class multivariate_categorical_distribution<double>;
  template class multivariate_categorical_distribution<float>;
}

using namespace libgm;

typedef table<double> table_type;
typedef multivariate_categorical_distribution<> dist_type;

std::size_t nsamples = 10000;
double tol = 0.02;

double marginal_diff(const dist_type& d, const table_type& t) {
  std::mt19937 rng;
  table_type estimate(t.shape(), 0.0);
  for (std::size_t i = 0; i < nsamples; ++i) {
    uint_vector sample = d(rng);
    BOOST_REQUIRE_EQUAL(sample.size(), t.arity());
    ++estimate(sample);
  }
  estimate /= double(nsamples);
  return std::inner_product(estimate.begin(), estimate.end(), t.begin(),
                            0.0, maximum<double>(), abs_difference<double>());
}

double conditional_diff(const dist_type& d, const table_type& t) {
  std::mt19937 rng;
  table_type estimate(t.shape(), 0.0);
  for (std::size_t last = 0; last < t.shape().back(); ++last) {
    uint_vector tail(1, last);
    for (std::size_t i = 0; i < nsamples; ++i) {
      uint_vector sample = d(rng, tail);
      BOOST_REQUIRE_EQUAL(sample.size(), t.arity() - 1);
      sample.push_back(last);
      ++estimate(sample);
    }
  }
  estimate /= double(nsamples);
  return std::inner_product(estimate.begin(), estimate.end(), t.begin(),
                            0.0, maximum<double>(), abs_difference<double>());
}

BOOST_AUTO_TEST_CASE(test_marginal) {
  table_type pt({2, 3}, {0.1, 0.05, 0.15, 0.25, 0.2, 0.25});
  table_type ct({2, 3});
  std::transform(pt.begin(), pt.end(), ct.begin(), logarithm<double>());
  BOOST_CHECK_SMALL(marginal_diff(dist_type(pt), pt), tol);
  BOOST_CHECK_SMALL(marginal_diff(dist_type(ct, log_tag()), pt), tol);
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  table_type pt({2, 3, 2},
                {0.1, 0.05, 0.15, 0.25, 0.2, 0.25,
                 0.15, 0.25, 0.05, 0.2, 0.3, 0.05});
  table_type ct({2, 3, 2});
  std::transform(pt.begin(), pt.end(), ct.begin(), logarithm<double>());
  BOOST_CHECK_SMALL(conditional_diff(dist_type(pt), pt), tol);
  BOOST_CHECK_SMALL(conditional_diff(dist_type(ct, log_tag()), pt), tol);
}
