#define BOOST_TEST_MODULE hybrid_assignment
#include <boost/test/unit_test.hpp>

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class hybrid_assignment<var>;
  template class hybrid_assignment<vec>;
}

using namespace libgm;

typedef hybrid_domain<var> domain_type;

BOOST_AUTO_TEST_CASE(test_all) {
  universe u;
  var x = var::continuous(u, "x");
  var y = var::continuous(u, "y");
  var z = var::discrete(u, "z", 4);
  var w = var::discrete(u, "w", 3);
  var q = var::continuous(u, "q");

  hybrid_assignment<var> a({{z, 1}, {w, 2}}, {{x, 3.0}, {y, 2.0}});

  // Container
  BOOST_CHECK_EQUAL(a.size(), 4);
  BOOST_CHECK(!a.empty());
  BOOST_CHECK(a == a);
  BOOST_CHECK(a != hybrid_assignment<var>(a.uint()));

  // UnorderedAssociativeContainer
  BOOST_CHECK_EQUAL(a.at(x), 3.0);
  BOOST_CHECK_EQUAL(a.at(y), 2.0);
  BOOST_CHECK_EQUAL(a.at(z), 1);
  BOOST_CHECK_EQUAL(a.at(w), 2);
  BOOST_CHECK_EQUAL(a[x], 3.0);
  BOOST_CHECK_EQUAL(a[y], 2.0);
  BOOST_CHECK_EQUAL(a[z], 1);
  BOOST_CHECK_EQUAL(a[w], 2);
  BOOST_CHECK_EQUAL(a.count(x), 1);
  BOOST_CHECK_EQUAL(a.count(z), 1);
  BOOST_CHECK_EQUAL(a.count(q), 0);
  BOOST_CHECK_EQUAL(a.erase(q), 0);
  BOOST_CHECK_EQUAL(a.erase(w), 1);

  // Assignment
  BOOST_CHECK_EQUAL(a.values({x, y, z}), hybrid_vector<>({1}, {3.0, 2.0}));
  BOOST_CHECK_EQUAL(a.values({y, x}), hybrid_vector<>({}, {2.0, 3.0}));
  BOOST_CHECK_EQUAL(a.insert({w, x}, hybrid_vector<>({1}, {0.5})), 1);
  BOOST_CHECK_EQUAL(a.insert_or_assign({w, q}, hybrid_vector<>({2}, {0.5})), 1);
  BOOST_CHECK(subset(domain_type({y, x, z}), a));
  BOOST_CHECK(!disjoint(domain_type({x, y, z}), a));

  // Clearing
  a.clear();
  BOOST_CHECK(a.empty());
}
