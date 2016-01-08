#define BOOST_TEST_MODULE logarithmic_vector_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/logarithmic_vector_ll.hpp>

namespace libgm {
  template class logarithmic_vector_ll<double>;
  template class logarithmic_vector_ll<float>;
}
