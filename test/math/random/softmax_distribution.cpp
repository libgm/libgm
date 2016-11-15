#define BOOST_TEST_MODULE softmax_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/softmax_distribution.hpp>

namespace libgm {
  template class softmax_distribution<double>;
  template class softmax_distribution<float>;
}

using namespace libgm;

std::size_t nsamples = 5000;
double tol = 0.03;

dense_vector<double>
sample(const softmax_distribution<double>& d,
       const dense_vector<double>& tail) {
  std::mt19937 rng;
  dense_vector<double> result(d.param().labels());
  result.fill(0.0);
  for (std::size_t i = 0; i < nsamples; ++i) {
    ++result[d(rng, tail)];
  }
  result /= double(nsamples);
  return result;
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  softmax_param<double> param(3, 2);
  param.bias() << 0.1, 0.2, 0.3;
  param.weight() << -0.1, 0.1, 0.2, 0.3, -0.2, 0.3;
  softmax_distribution<double> distribution(param);
  for (double a = -2.0; a <= 2.0; a += 0.5) {
    double b = 1.0;
    dense_vector<double> tail(2);
    tail[0] = a;
    tail[1] = b;
    dense_vector<double> result(3);
    result[0] = std::exp(0.1 - 0.1*a + 0.1*b);
    result[1] = std::exp(0.2 + 0.2*a + 0.3*b);
    result[2] = std::exp(0.3 - 0.2*a + 0.3*b);
    result /= result.sum();
    dense_vector<double> estimate = sample(distribution, tail);
    BOOST_CHECK_SMALL((result - estimate).cwiseAbs().sum(), tol);
  }
}
