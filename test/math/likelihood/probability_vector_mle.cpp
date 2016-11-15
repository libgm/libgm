#define BOOST_TEST_MODULE probability_vector_mle
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_vector_mle.hpp>
#include <libgm/math/likelihood/probability_vector_ll.hpp>
#include <libgm/math/random/categorical_distribution.hpp>

#include "test_mle.hpp"

namespace libgm {
  template class probability_vector_mle<double>;
  template class probability_vector_mle<float>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_reconstruction) {
  dense_vector<double> param(4);
  param << 0.1, 0.4, 0.3, 0.2;
  double diff = reconstruction_error<
    categorical_distribution<>,
    probability_vector_mle<>,
    probability_vector_ll<>
      > (10000, param, param.size());
  BOOST_CHECK_SMALL(diff, 0.01);
}
