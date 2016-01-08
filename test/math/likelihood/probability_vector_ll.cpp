#define BOOST_TEST_MODULE probability_vector_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_vector_ll.hpp>

namespace libgm {
  template class probability_vector_ll<double>;
  template class probability_vector_ll<float>;
}
