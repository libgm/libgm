#define BOOST_TEST_MODULE commutative_semiring
#include <boost/test/unit_test.hpp>

#include <libgm/factor/util/commutative_semiring.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/util/operations.hpp>

#include <random>
#include <vector>

namespace libgm {
  template class sum_product<ptable>;
  template class max_product<ptable>;
  template class min_sum<ptable>;
  template class max_sum<ptable>;
  
  template class sum_product<cgaussian>;
  template class max_product<cgaussian>;
}

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_ops) {
  size_t nvars = 3;
  size_t arity = 2;

  universe u;
  domain vars = u.new_finite_variables(nvars, "x", arity);
  variable v = vars[0];

  uniform_table_generator<ptable> gen;
  std::mt19937 rng;
  std::vector<ptable> f;
  for (size_t i = 0; i < 3; ++i) {
    f.push_back(gen(vars, rng));
  }
  
  libgm::sum_product<ptable> sum_product;
  libgm::max_product<ptable> max_product;
  libgm::min_sum<ptable> min_sum;
  libgm::max_sum<ptable> max_sum;
  
  BOOST_CHECK_EQUAL(combine_all(f, sum_product), f[0] * f[1] * f[2]);
  BOOST_CHECK_EQUAL(combine_all(f, max_product), f[0] * f[1] * f[2]);
  // BOOST_CHECK_EQUAL(combine_all(f, min_sum), f[0] + f[1] + f[2]);
  // BOOST_CHECK_EQUAL(combine_all(f, max_sum), f[0] + f[1] + f[2]);

  BOOST_CHECK_EQUAL(sum_product.combine(f[0], f[1]), f[0] * f[1]);
  BOOST_CHECK_EQUAL(max_product.combine(f[0], f[1]), f[0] * f[1]);
  //BOOST_CHECK_EQUAL(min_sum.combine(f[0], f[1]), f[0] + f[1]);
  //BOOST_CHECK_EQUAL(max_sum.combine(f[0], f[1]), f[0] + f[1]);
  
  BOOST_CHECK_EQUAL(sum_product.collapse(f[0], {v}), f[0].marginal({v}));
  BOOST_CHECK_EQUAL(max_product.collapse(f[0], {v}), f[0].maximum({v}));
  //BOOST_CHECK_EQUAL(min_sum.collapse(f[0], {v}), f[0].minimum({v}));
  //BOOST_CHECK_EQUAL(max_sum.collapse(f[0], {v}), f[0].maximum({v}));
}
