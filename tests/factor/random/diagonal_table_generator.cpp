#define BOOST_TEST_MODULE diagonal_table_generator
#include <boost/test/unit_test.hpp>

#include <libgm/factor/random/diagonal_table_generator.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/factor/canonical_table.hpp>
#include <libgm/factor/probability_table.hpp>

#include <boost/mpl/list.hpp>

using namespace libgm;

typedef canonical_table<var> ctable;
typedef probability_table<var> ptable;

namespace libgm {
  template class diagonal_table_generator<ctable>;
  template class diagonal_table_generator<ptable>;
}

std::size_t nsamples = 1000;
const double lower = -0.7;
const double upper = +0.5;

typedef boost::mpl::list<ctable, ptable> factor_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_all, F, factor_types) {
  universe u;
  var x = var::discrete(u, "x", 3);
  var y = var::discrete(u, "y", 3);
  domain<var> xy = {x, y};

  std::mt19937 rng;
  diagonal_table_generator<F> gen(lower, upper);

  // check the marginals
  double sum = 0.0;
  std::size_t count = 0;
  uint_vector shape(2, 3);
  for (std::size_t i = 0; i < nsamples; ++i) {
    F f = gen(xy, rng);
    uint_vector_iterator it(shape), end(2);
    for (; it != end; ++it) {
      const uint_vector& index = *it;
      if (index[0] == index[1]) {
        BOOST_CHECK(f.param(index) >= lower && f.param(index) <= upper);
        sum += f.param(index);
        ++count;
      } else {
        BOOST_CHECK_SMALL(log(f(index)), 1e-8);
      }
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum / count, (lower+upper)/2, 0.05);
}
