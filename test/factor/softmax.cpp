#define BOOST_TEST_MODULE softmax
#include <boost/test/unit_test.hpp>

#include <libgm/factor/softmax.hpp>

namespace libgm {
  template class softmax<double, variable>;
  template class softmax<float, variable>;
}

using namespace libgm;

