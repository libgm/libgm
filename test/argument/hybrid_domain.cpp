#define BOOST_TEST_MODULE hybrid_domain
#include <boost/test/unit_test.hpp>

#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

namespace libgm {
  template class hybrid_domain<var>;
  template class hybrid_domain<vec>;
}

using namespace libgm;

typedef hybrid_domain<var> domain_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  var x = var::continuous(u, "x");
  var y = var::continuous(u, "y");
  var z = var::discrete(u, "z", 2);
  var w = var::discrete(u, "w", 3);

  domain_type d({z, w}, {x, y});
  domain_type e({x, y, z, w});

  BOOST_CHECK_EQUAL(d, e);
  BOOST_CHECK_EQUAL(d.discrete(), domain<var>({z, w}));
  BOOST_CHECK_EQUAL(d.continuous(), domain<var>({x, y}));
}

BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  var x = var::continuous(u, "x");
  var y = var::continuous(u, "y");
  var z = var::discrete(u, "z", 2);
  var w = var::discrete(u, "w", 3);
  var q = var::continuous(u, "q");
  var r = var::discrete(u, "r", 2);

  domain_type d({z, w}, {x, y});
  domain_type e;
  domain_type f({z, w});

  // Container
  BOOST_CHECK_EQUAL(d.size(), 4);
  BOOST_CHECK_EQUAL(e.size(), 0);
  BOOST_CHECK_EQUAL(f.size(), 2);
  BOOST_CHECK(!d.empty());
  BOOST_CHECK(e.empty());
  BOOST_CHECK(!f.empty());

  // Mutations
  swap(e, f);
  BOOST_CHECK_EQUAL(e.size(), 2);
  BOOST_CHECK_EQUAL(f.size(), 0);
  e.clear();
  BOOST_CHECK_EQUAL(e.size(), 0);

  f.push_back(q);
  f.push_back(q);
  f.push_back(r);
  BOOST_CHECK_EQUAL(f.discrete().size(), 1);
  BOOST_CHECK_EQUAL(f.continuous().size(), 2);
  f.unique();
  BOOST_CHECK_EQUAL(f.discrete().size(), 1);
  BOOST_CHECK_EQUAL(f.continuous().size(), 1);

  // Sequence
  BOOST_CHECK(d.prefix({x, y, z}));
  BOOST_CHECK(d.suffix({y, w}));
  BOOST_CHECK_EQUAL(concat(d, f), domain_type({x, y, z, w, q, r}));

  // Set
  BOOST_CHECK_EQUAL(d.count(x), 1);
  BOOST_CHECK_EQUAL(d.count(w), 1);
  BOOST_CHECK_EQUAL(d.count(q), 0);
  BOOST_CHECK_EQUAL(d + f, domain_type({x, y, z, w, q, r}));
  BOOST_CHECK_EQUAL(d - domain_type({y, z}), domain_type({x, w}));
  BOOST_CHECK_EQUAL(d & domain_type({y, z, q}), domain_type({y, z}));
  BOOST_CHECK(disjoint(d, domain_type({q, r})));
  BOOST_CHECK(!disjoint(d, domain_type({q, r, x})));
  BOOST_CHECK(equivalent(d, domain_type({w, z, y, x})));
  BOOST_CHECK(!equivalent(d, domain_type({x, y})));
  BOOST_CHECK(subset(domain_type({y, x}), d));
  BOOST_CHECK(!subset(domain_type({x, q}), d));
  BOOST_CHECK(superset(d, domain_type({y, x})));
  BOOST_CHECK(!superset(domain_type({x, q}), d));
  BOOST_CHECK(compatible(domain_type({x, y, z}), domain_type({y, q, r})));
  BOOST_CHECK_EQUAL(d.num_values(), 6);
  BOOST_CHECK_EQUAL(d.num_dimensions(), 4);
}
