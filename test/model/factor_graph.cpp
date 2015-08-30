#define BOOST_TEST_MODULE factor_graph
#include <boost/test/unit_test.hpp>

#include <libgm/model/factor_graph.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>

#include <random>

#include "predicates.hpp"

using namespace libgm;

typedef canonical_gaussian<var> cgaussian;
typedef probability_table<var> ptable;

namespace libgm {
  template class factor_graph<cgaussian>;
  template class factor_graph<ptable>;
}

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<domain<var> >);

struct fixture {
  typedef factor_graph<ptable> model_type;

  fixture()
    : nvars(10) {
    // Random number generator
    std::mt19937 rng;
    uniform_table_generator<ptable> gen;

    // Create some variables
    for(std::size_t i = 0; i < nvars; ++i) {
      x.push_back(var::discrete(u, "x" + std::to_string(i), 2));
    }

    // Create some unary factors
    for(std::size_t i = 0; i < nvars; ++i) {
      fg.add_factor(gen({x[i]}, rng));
    }

    // For every two variables in a chain create a factor
    for(std::size_t i = 0; i < x.size() - 1; ++i) {
      fg.add_factor(gen({x[i], x[i+1]}, rng));
    }
  }

  universe u;
  std::size_t nvars;
  domain<var> x;
  model_type fg;
};

BOOST_FIXTURE_TEST_CASE(test_structure, fixture) {
  for (std::size_t i = 0; i < nvars; ++i) {
    std::vector<domain<var> > args1;
    args1.push_back({x[i]});
    if (i > 0) {
      args1.push_back({x[i-1], x[i]});
    }
    if (i < nvars - 1) {
      args1.push_back({x[i], x[i+1]});
    }

    std::vector<domain<var> > args2;
    for (libgm::id_t id : fg.neighbors(x[i])) {
      args2.push_back(fg.arguments(id));
    }

    std::sort(args1.begin(), args1.end());
    std::sort(args2.begin(), args2.end());
    BOOST_CHECK_EQUAL(args1, args2);
  }
  BOOST_CHECK_EQUAL(fg.num_arguments(), nvars);
  BOOST_CHECK_EQUAL(fg.num_factors(), 2*nvars - 1);
  BOOST_CHECK_EQUAL(fg.num_edges(), 2*(nvars-1) + nvars);
}

BOOST_FIXTURE_TEST_CASE(test_simplify, fixture) {
  fg.simplify();
  for (std::size_t i = 0; i < nvars; ++i) {
    std::vector<domain<var> > args1;
    if (i > 0) {
      args1.push_back({x[i-1], x[i]});
    }
    if (i < nvars - 1) {
      args1.push_back({x[i], x[i+1]});
    }

    std::vector<domain<var> > args2;
    for (libgm::id_t id : fg.neighbors(x[i])) {
      args2.push_back(fg.arguments(id));
    }

    std::sort(args1.begin(), args1.end());
    std::sort(args2.begin(), args2.end());
    BOOST_CHECK_EQUAL(args1, args2);
  }
  BOOST_CHECK_EQUAL(fg.num_arguments(), nvars);
  BOOST_CHECK_EQUAL(fg.num_factors(), nvars - 1);
  BOOST_CHECK_EQUAL(fg.num_edges(), 2*(nvars-1));
}

/*
BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(fg, u));
}
*/
