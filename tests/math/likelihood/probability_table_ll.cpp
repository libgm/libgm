#define BOOST_TEST_MODULE probability_table_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/probability_table_ll.hpp>

namespace libgm {
  template class probability_table_ll<double>;
  template class probability_table_ll<float>;
}
