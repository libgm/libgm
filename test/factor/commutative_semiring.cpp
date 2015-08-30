#define BOOST_TEST_MODULE commutative_semiring
#include <boost/test/unit_test.hpp>

#include <libgm/factor/util/commutative_semiring.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/factor/util/operations.hpp>

#include <random>
#include <vector>


using namespace libgm;
typedef canonical_gaussian<var> cgaussian;
typedef probability_table<var> ptable;

namespace libgm {
  template class sum_product<ptable>;
  template class max_product<ptable>;
  template class min_sum<ptable>;
  template class max_sum<ptable>;

  template class sum_product<cgaussian>;
  template class max_product<cgaussian>;
}

BOOST_AUTO_TEST_CASE(test_ops) {
  std::size_t nvars = 3;
  std::size_t arity = 2;

  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 1);

  uniform_table_generator<ptable> gen;
  std::mt19937 rng;
  std::vector<ptable> f;
  for (std::size_t i = 0; i < 3; ++i) {
    f.push_back(gen({x, y}, rng));
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

  BOOST_CHECK_EQUAL(sum_product.collapse(f[0], {y}), f[0].marginal({y}));
  BOOST_CHECK_EQUAL(max_product.collapse(f[0], {y}), f[0].maximum({y}));
  //BOOST_CHECK_EQUAL(min_sum.collapse(f[0], {y}), f[0].minimum({v}));
  //BOOST_CHECK_EQUAL(max_sum.collapse(f[0], {y}), f[0].maximum({v}));
}
