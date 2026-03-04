#define BOOST_TEST_MODULE probability_matrix_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_matrix_ll.hpp>

namespace libgm {
  template class probability_matrix_ll<double>;
  template class probability_matrix_ll<float>;
}
