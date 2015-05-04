#define BOOST_TEST_MODULE array_distribution
#include <boost/test/unit_test.hpp>

#include <libgm/math/random/array_distribution.hpp>

namespace libgm {
  template class array_distribution<double, 1>;
  template class array_distribution<double, 2>;
  template class array_distribution<float, 1>;
  template class array_distribution<float, 2>;
}

using namespace libgm;

typedef array_distribution<double, 1> dist1_type;
typedef array_distribution<double, 2> dist2_type;
typedef dist1_type::param_type array1_type;
typedef dist2_type::param_type array2_type;

size_t nsamples = 10000;
double tol = 0.01;

double marginal_diff(const dist1_type& d, const array1_type& a) {
  std::mt19937 rng;
  array1_type estimate = array1_type::Zero(a.size());
  for (size_t i = 0; i < nsamples; ++i) {
    ++estimate[d(rng)];
  }
  estimate /= nsamples;
  return abs(estimate - a).maxCoeff();
}

double marginal_diff(const dist2_type& d, const array2_type& a) {
  std::mt19937 rng;
  array2_type estimate = array2_type::Zero(a.rows(), a.cols());
  for (size_t i = 0; i < nsamples; ++i) {
    finite_index sample = d(rng);
    BOOST_REQUIRE_EQUAL(sample.size(), 2);
    ++estimate(sample[0], sample[1]);
  }
  estimate /= nsamples;
  return abs(estimate - a).maxCoeff();
}

double conditional_diff(const dist2_type& d, const array2_type& a) {
  std::mt19937 rng;
  array2_type estimate = array2_type::Zero(a.rows(), a.cols());
  for (size_t tail = 0; tail < a.cols(); ++tail) {
    for (size_t i = 0; i < nsamples; ++i) {
      ++estimate(d(rng, tail), tail);
    }
  }
  estimate /= nsamples;
  return abs(estimate - a).maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_marginal1) {
  array1_type a(4);
  a << 0.2, 0.1, 0.4, 0.3;
  BOOST_CHECK_SMALL(marginal_diff(dist1_type(a), a), tol);
  BOOST_CHECK_SMALL(marginal_diff(dist1_type(log(a), log_tag()), a), tol);
}

BOOST_AUTO_TEST_CASE(test_marginal2) {
  array2_type a(2, 3);
  a << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  BOOST_CHECK_SMALL(marginal_diff(dist2_type(a), a), tol);
  BOOST_CHECK_SMALL(marginal_diff(dist2_type(log(a), log_tag()), a), tol);
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  array2_type a(3, 2);
  a << 0.1, 0.25, 0.4, 0.60, 0.5, 0.15;
  BOOST_CHECK_SMALL(conditional_diff(dist2_type(a), a), tol);
  BOOST_CHECK_SMALL(conditional_diff(dist2_type(log(a), log_tag()), a), tol);
}
