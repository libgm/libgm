#define BOOST_TEST_MODULE multinomial_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/multinomial_distribution.hpp>

#include <random>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_sampling) {
  std::mt19937 rng;

  Eigen::Array4d p = {0.2, 0.2, 0.5, 0.1};
  std::vector<double> count(4);
  multinomial_distribution<double> dist(p);

  std::size_t nsamples = 200000;
  for(std::size_t i = 0; i < nsamples; i++) {
    count[dist(rng)] += 1.0 / nsamples;
  }

  for (std::size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(p[i], count[i], 1);
  }
}
