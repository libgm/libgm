#define BOOST_TEST_MODULE probability_matrix_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_matrix_mle.hpp>
#include <libgm/math/likelihood/probability_matrix_ll.hpp>
#include <libgm/math/random/bivariate_categorical_distribution.hpp>

#include "test_mle.hpp"

namespace libgm {
  template class probability_matrix_mle<double>;
  template class probability_matrix_mle<float>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_reconstruction) {
  dense_matrix<double> param(2, 3);
  param << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  double diff = reconstruction_error<
    bivariate_categorical_distribution<>,
    probability_matrix_mle<>,
    probability_matrix_ll<>
      > (10000, param, std::make_pair(2, 3));
  BOOST_CHECK_SMALL(diff, 0.01);
}
