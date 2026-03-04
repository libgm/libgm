#define BOOST_TEST_MODULE softmax
#include <boost/test/unit_test.hpp>

#include <libgm/factor/softmax.hpp>

#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class softmax<var>;
  template class softmax<vec>;
}

using namespace libgm;

