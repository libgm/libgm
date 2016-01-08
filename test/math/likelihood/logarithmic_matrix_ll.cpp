#define BOOST_TEST_MODULE logarithmic_matrix_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/logarithmic_matrix_ll.hpp>

namespace libgm {
  template class logarithmic_matrix_ll<double>;
  template class logarithmic_matrix_ll<float>;
}
