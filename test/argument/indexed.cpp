#define BOOST_TEST_MODULE sequence
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class indexed<var>;
  template class indexed<vec>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::nullptr_t);

BOOST_AUTO_TEST_CASE(test_var) {
  universe u;
  var p = var::discrete(u, "p", 4);
  var q = var::discrete(u, "q", 2);
  indexed<p> pi = p(2);
  indexed<p> pj = p(3);
  indexed<q> qk = q(-1);

  // Accessors
  BOOST_CHECK_EQUAL(pi.pair(), std::make_pair(p, 2));
  BOOST_CHECK_EQUAL(qk.pair(), std::make_pair(q, -1));
  BOOST_CHECK_EQUAL(pi.process(), p);
  BOOST_CHECK_EQUAL(pi.index(), 2);

  // Comparisons
  BOOST_CHECK(pi == pi);
  BOOST_CHECK(pi <= pi);
  BOOST_CHECK(pj >= pi);
  BOOST_CHECK(!(pi < pi));
  BOOST_CHECK(!(pj > pi));

  BOOST_CHECK(pi != pj);
  BOOST_CHECK(pi <= pj);
  BOOST_CHECK(pj >= pi);
  BOOST_CHECK(pi < pj);
  BOOST_CHECK(pj > pi);

  BOOST_CHECK(pi != qk);
  BOOST_CHECK((pi < qk) ^ (qk < pi));
  BOOST_CHECK((pi > qk) ^ (qk > pi));

  // Traits
  BOOST_CHECK_EQUAL(pi.arity(), 1);
  BOOST_CHECK_EQUAL(pi.num_values(), 4);
  BOOST_CHECK_EQUAL(qi.num_values(), 2);
  BOOST_CHECK(pi.discrete());
  BOOST_CHECK(!pi.continuous());
}

BOOST_AUTO_TEST_CASE(test_vec) {
  universe u;
  vec p = vec::continuous(u, "p", 4);
  vec q = vec::continuous(u, "q", 2);

  // Accessors
  BOOST_CHECK_EQUAL(pi.pair(), std::make_pair(p, 2));
  BOOST_CHECK_EQUAL(qk.pair(), std::make_pair(q, -1));
  BOOST_CHECK_EQUAL(pi.process(), p);
  BOOST_CHECK_EQUAL(pi.index(), 2);

  // Comparisons
  BOOST_CHECK(pi == pi);
  BOOST_CHECK(pi <= pi);
  BOOST_CHECK(pj >= pi);
  BOOST_CHECK(!(pi < pi));
  BOOST_CHECK(!(pj > pi));

  BOOST_CHECK(pi != pj);
  BOOST_CHECK(pi <= pj);
  BOOST_CHECK(pj >= pi);
  BOOST_CHECK(pi < pj);
  BOOST_CHECK(pj < pi);

  BOOST_CHECK(pi != qk);
  BOOST_CHECK((pi < qk) ^ (qk < pi));
  BOOST_CHECK((pi > qk) ^ (qk > pi));

  // Traits
  BOOST_CHECK_EQUAL(pi.arity(), 4);
  BOOST_CHECK(!pi.discrete());
  BOOST_CHECK(pi.continuous());
}
