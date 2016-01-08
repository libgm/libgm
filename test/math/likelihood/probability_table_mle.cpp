#define BOOST_TEST_MODULE probability_table_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_table_mle.hpp>
#include <libgm/math/likelihood/probability_table_ll.hpp>
#include <libgm/math/random/multivariate_categorical_distribution.hpp>

#include "test_mle.hpp"

namespace libgm {
  template class probability_table_mle<double>;
  template class probability_table_mle<float>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_reconstruction) {
  table<double> param({2, 3}, {0.1, 0.05, 0.15, 0.25, 0.2, 0.25});
  double diff = reconstruction_error<
    multivariate_categorical_distribution<>,
    probability_table_mle<>,
    probability_table_ll<>
      > (10000, param, param.shape());
  BOOST_CHECK_SMALL(diff, 0.01);
}
