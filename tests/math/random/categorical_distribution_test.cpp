#define BOOST_TEST_MODULE categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/categorical_distribution.hpp>
#include <libgm/factor/logarithmic_vector.hpp>
#include <libgm/factor/probability_vector.hpp>

using namespace libgm;

using Dist = CategoricalDistribution<double>;
using Vec = Vector<double>;
using PVec = ProbabilityVector<double>;
using LVec = LogarithmicVector<double>;

std::size_t nsamples = 10000;
double tol = 0.01;

double marginal_diff(const Dist& d, const Vec& v) {
  std::mt19937 rng;
  Vec estimate = Vec::Zero(v.size());
  for (std::size_t i = 0; i < nsamples; ++i) {
    ++estimate[d(rng)];
  }
  estimate /= double(nsamples);
  return (estimate - v).array().abs().maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_marginal1) {
  Vec v(4);
  v << 0.2, 0.1, 0.4, 0.3;
  BOOST_CHECK_SMALL(marginal_diff(Dist(PVec(v)), v), tol);
  BOOST_CHECK_SMALL(marginal_diff(Dist(LVec(v.array().log())), v), tol);
}
