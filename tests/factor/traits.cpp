#define BOOST_TEST_MODULE factor_traits
#include <boost/test/unit_test.hpp>

#include <libgm/factor/utility/traits.hpp>

#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/moment_gaussian.hpp>

using namespace libgm;

using cgaussian = canonical_gaussian<>;
using mgaussian = moment_gaussian<>;

BOOST_AUTO_TEST_CASE(test_operators) {
  BOOST_CHECK(!has_plus<cgaussian>::value);
  BOOST_CHECK(!has_minus<cgaussian>::value);
  BOOST_CHECK(has_multiplies<cgaussian>::value);
  BOOST_CHECK(has_divides<cgaussian>::value);

  BOOST_CHECK(!(has_plus<cgaussian, logd>::value));
  BOOST_CHECK(!(has_minus<cgaussian, logd>::value));
  BOOST_CHECK((has_multiplies<cgaussian, logd>::value));
  BOOST_CHECK((has_multiplies<logd, cgaussian>::value));
  BOOST_CHECK((has_divides<cgaussian, logd>::value));
  BOOST_CHECK((has_divides<logd, cgaussian>::value));

  BOOST_CHECK(!has_plus_assign<cgaussian>::value);
  BOOST_CHECK(!has_minus_assign<cgaussian>::value);
  BOOST_CHECK(has_multiplies_assign<cgaussian>::value);
  BOOST_CHECK(has_divides_assign<cgaussian>::value);

  BOOST_CHECK(!(has_plus_assign<cgaussian, logd>::value));
  BOOST_CHECK(!(has_minus_assign<cgaussian, logd>::value));
  BOOST_CHECK((has_multiplies_assign<cgaussian, logd>::value));
  BOOST_CHECK((has_divides_assign<cgaussian, logd>::value));
}

BOOST_AUTO_TEST_CASE(test_functions) {
  BOOST_CHECK(!has_max<cgaussian>::value);
  BOOST_CHECK(!has_min<cgaussian>::value);
  BOOST_CHECK(has_weighted_update<cgaussian>::value);
  BOOST_CHECK(!has_cross_entropy<cgaussian>::value);
  BOOST_CHECK(has_kl_divergence<cgaussian>::value);
  BOOST_CHECK(!has_js_divergence<cgaussian>::value);
  BOOST_CHECK(has_max_diff<cgaussian>::value);
}

BOOST_AUTO_TEST_CASE(test_members) {
  BOOST_CHECK(!has_head<cgaussian>::value);
  BOOST_CHECK(!has_tail<cgaussian>::value);
  BOOST_CHECK(has_head<mgaussian>::value);
  BOOST_CHECK(has_tail<mgaussian>::value);
  BOOST_CHECK(has_marginal<cgaussian>::value);
  BOOST_CHECK((has_marginal<cgaussian, void>::value));
  BOOST_CHECK(has_maximum<cgaussian>::value);
  BOOST_CHECK((has_maximum<cgaussian, void>::value));
  BOOST_CHECK(!has_minimum<cgaussian>::value);
  BOOST_CHECK(has_conditional<cgaussian>::value);
  BOOST_CHECK(has_restrict<cgaussian>::value);
  BOOST_CHECK(!(has_sample<cgaussian, std::mt19937>::value));
  BOOST_CHECK(has_entropy<cgaussian>::value);
  BOOST_CHECK((has_entropy<cgaussian, void>::value));
  BOOST_CHECK(has_mutual_information<cgaussian>::value);
}
