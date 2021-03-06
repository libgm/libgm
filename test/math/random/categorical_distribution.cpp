#define BOOST_TEST_MODULE categorical_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/categorical_distribution.hpp>

namespace libgm {
  template class categorical_distribution<double>;
  template class categorical_distribution<float>;
}

using namespace libgm;

typedef categorical_distribution<> dist_type;

std::size_t nsamples = 10000;
double tol = 0.01;

double marginal_diff(const dist_type& d, const dense_vector<>& v) {
  std::mt19937 rng;
  dense_vector<> estimate = dense_vector<>::Zero(v.size());
  for (std::size_t i = 0; i < nsamples; ++i) {
    ++estimate[d(rng)];
  }
  estimate /= double(nsamples);
  return (estimate - v).array().abs().maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_marginal1) {
  dense_vector<> v(4);
  v << 0.2, 0.1, 0.4, 0.3;
  BOOST_CHECK_SMALL(marginal_diff(dist_type(v, prob_tag()), v), tol);
  BOOST_CHECK_SMALL(marginal_diff(dist_type(v.array().log(), log_tag()), v), tol);
}
