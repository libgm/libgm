#define BOOST_TEST_MODULE moment_gaussian_generator
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>

namespace libgm {
  template class moment_gaussian_generator<double>;
  template class moment_gaussian_generator<float>;
}

using namespace libgm;

typedef real_vector<double> vec_type;
typedef real_matrix<double> mat_type;

std::size_t nsamples = 100;

BOOST_AUTO_TEST_CASE(test_all) {
  universe u;
  variable x1 = u.new_continuous_variable("x1", 1);
  variable x2 = u.new_continuous_variable("x2", 2);
  variable y = u.new_continuous_variable("y", 1);
  domain xs = {x1, x2};
  domain ys = {y};
  domain xy = {x1, x2, y};

  std::mt19937 rng;
  moment_gaussian_generator<double> gen(-0.5, 1.5, 2.0, 0.3, 0);
  
  // test marginals
  double sum = 0.0;
  for (std::size_t i = 0; i < nsamples; ++i) {
    mgaussian mg = gen(xy, rng);
    const vec_type& mean = mg.mean();
    const mat_type& cov = mg.covariance();
    BOOST_CHECK(mg.is_marginal());
    BOOST_CHECK_EQUAL(mean.rows(), 4);
    BOOST_CHECK_EQUAL(cov.rows(), 4);
    BOOST_CHECK_EQUAL(cov.cols(), 4);
    BOOST_CHECK((mean.array() >= -0.5 && mean.array() <= 1.5).all());
    sum += mean.sum();
    for (std::size_t r = 0; r < 4; ++r) {
      for (std::size_t c = 0; c < 4; ++c) {
        if (r == c) {
          BOOST_CHECK_CLOSE(cov(r, c), 2.0, 1e-10);
        } else {
          BOOST_CHECK_CLOSE(cov(r, c), 0.6, 1e-10);
        }
      }
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum / nsamples / 4, 0.5, 0.05);
  
  // test conditionals
  double sum_mean = 0.0;
  double sum_coef = 0.0;
  for (std::size_t i = 0; i < nsamples; ++i) {
    mgaussian mg = gen(ys, xs, rng);
    const vec_type& mean = mg.mean();
    const mat_type& cov  = mg.covariance();
    const mat_type& coef = mg.coefficients();
    BOOST_CHECK(!mg.is_marginal());
    BOOST_CHECK_EQUAL(mg.head_size(), 1);
    BOOST_CHECK_EQUAL(mg.tail_size(), 3);
    BOOST_CHECK((mean.array() >= -0.5 && mean.array() <= 1.5).all());
    BOOST_CHECK((coef.array() >=  0.0 && coef.array() <= 1.0).all());
    sum_mean += mean.sum();
    sum_coef += coef.sum();
  }
  BOOST_CHECK_CLOSE_FRACTION(sum_mean / nsamples, 0.5, 0.05);
  BOOST_CHECK_CLOSE_FRACTION(sum_coef / nsamples / 3, 0.5, 0.05);
}
