#define BOOST_TEST_MODULE canonical_array_ll
#include <boost/test/unit_test.hpp>

#include <libgm/math/likelihood/canonical_array_ll.hpp>

namespace libgm {
  template class canonical_array_ll<double, 1>;
  template class canonical_array_ll<double, 2>;
  template class canonical_array_ll<float, 1>;
  template class canonical_array_ll<float, 2>;
}
