#define BOOST_TEST_MODULE sequence
#include <boost/test/unit_test.hpp>

#include <libgm/argument/sequence.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class field<var, std::size_t>;
  template class field<vec, std::size_t>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::nullptr_t);

BOOST_AUTO_TEST_CASE(test_var) {
  universe u;
  sequence<var> p = var::discrete(u, "p", 4).desc();
  sequence<var> q = var::discrete(u, "q", 2).desc();

  // Accessors
  BOOST_CHECK_EQUAL(p.desc()->name, "p");
  BOOST_CHECK_EQUAL(p.desc()->cardinality, 4);
  BOOST_CHECK_EQUAL(sequence<var>().desc(), nullptr);
  BOOST_CHECK_EQUAL(p(10).index(), 10);

  // Comparisons
  BOOST_CHECK(p == p);
  BOOST_CHECK(p != q);
  BOOST_CHECK(p <= p);
  BOOST_CHECK(p >= p);
  BOOST_CHECK((p < q) ^ (q < p));
  BOOST_CHECK((p > q) ^ (q > p));

  // Traits
  BOOST_CHECK_EQUAL(p.num_dimensions(), 1);
  BOOST_CHECK_EQUAL(p.num_values(), 4);
  BOOST_CHECK_EQUAL(q.num_values(), 2);
  BOOST_CHECK(p.discrete());
  BOOST_CHECK(!p.continuous());
}

BOOST_AUTO_TEST_CASE(test_vec) {
  universe u;
  sequence<vec> p = vec::continuous(u, "p", 4).desc();
  sequence<vec> q = vec::continuous(u, "q", 2).desc();

  // Accessors
  BOOST_CHECK_EQUAL(p.desc()->name, "p");
  BOOST_CHECK_EQUAL(p.desc()->length, 4);
  BOOST_CHECK_EQUAL(sequence<vec>().desc(), nullptr);
  BOOST_CHECK_EQUAL(p(10).index(), 10);

  // Comparisons
  BOOST_CHECK(p == p);
  BOOST_CHECK(p != q);
  BOOST_CHECK(p <= p);
  BOOST_CHECK(p >= p);
  BOOST_CHECK((p < q) ^ (q < p));
  BOOST_CHECK((p > q) ^ (q > p));

  // Traits
  BOOST_CHECK_EQUAL(p.num_dimensions(), 4);
  BOOST_CHECK(!p.discrete());
  BOOST_CHECK(p.continuous());
}
