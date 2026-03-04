#define BOOST_TEST_MODULE alternating_generator
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/alternating_generator.hpp>
#include <libgm/factor/random/moment_gaussian_generator.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>

#include <random>

using namespace libgm;

typedef probability_table<var> ptable;
typedef moment_gaussian<var> mgaussian;

namespace libgm{
  template class alternating_generator<uniform_table_generator<ptable> >;
  template class alternating_generator<moment_gaussian_generator<var> >;
}

BOOST_AUTO_TEST_CASE(test_constructors) {
  uniform_table_generator<ptable> def_gen(1.0, 2.0);
  uniform_table_generator<ptable> alt_gen(3.0, 4.0);
  uniform_table_generator<ptable>::param_type def_par = def_gen.param();
  uniform_table_generator<ptable>::param_type alt_par = alt_gen.param();


  alternating_generator<uniform_table_generator<ptable>> gen1(def_gen, alt_gen, 1);
  alternating_generator<uniform_table_generator<ptable>> gen2(def_par, alt_par, 3);

  BOOST_CHECK_EQUAL(gen1.param().def_param.lower, 1.0);
  BOOST_CHECK_EQUAL(gen1.param().alt_param.lower, 3.0);
  BOOST_CHECK_EQUAL(gen2.param().def_param.lower, 1.0);
  BOOST_CHECK_EQUAL(gen2.param().alt_param.lower, 3.0);
  BOOST_CHECK_EQUAL(gen1.param().period, 1);
  BOOST_CHECK_EQUAL(gen2.param().period, 3);
}

BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  var x = var::discrete(u, "x", 2);
  var y = var::discrete(u, "y", 1);
  domain<var> xs = { x };
  domain<var> ys = { y };
  domain<var> xy = { x, y };

  uniform_table_generator<ptable> def_gen(-2.0, -1.0); // log space
  uniform_table_generator<ptable> alt_gen(+1.0, +2.0); // log space

  alternating_generator<uniform_table_generator<ptable>> gen(def_gen, alt_gen, 3);
  std::mt19937 rng;

  // test marginals
  BOOST_CHECK_LT(gen(xy, rng)[0], 1.0);
  BOOST_CHECK_LT(gen(xy, rng)[0], 1.0);
  BOOST_CHECK_GT(gen(xy, rng)[0], 1.0);
  BOOST_CHECK_LT(gen(xy, rng)[0], 1.0);
  BOOST_CHECK_LT(gen(xy, rng)[0], 1.0);
  BOOST_CHECK_GT(gen(xy, rng)[0], 1.0);

  // test conditionals
  // not sure what to do besides instantiating the operator
  gen(ys, xs, rng);
  gen(ys, xs, rng);
  gen(ys, xs, rng);
  gen(ys, xs, rng);
}
