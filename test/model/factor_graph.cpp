#define BOOST_TEST_MODULE factor_graph
#include <boost/test/unit_test.hpp>

#include <libgm/model/factor_graph.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>

#include <boost/range/algorithm.hpp>

#include <random>

#include "predicates.hpp"

namespace libgm {
  template class factor_graph<cgaussian>;
  template class factor_graph<ptable>;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<domain>);

struct fixture {
  typedef factor_graph<ptable> model_type;

  fixture()
    : nvars(10) {
    // Random number generator
    std::mt19937 rng;
    uniform_table_generator<ptable> gen;

    // Create some variables
    x.resize(nvars);
    for(std::size_t i = 0; i < nvars; ++i) {
      x[i] = u.new_finite_variable("Variable: " + std::to_string(i), 2);
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
  std::vector<variable> x;
  model_type fg;
};

struct domain_less {
  template <typename Set>
  bool operator()(const Set& a, const Set& b) {
    return boost::lexicographical_compare(a, b);
  }
};

BOOST_FIXTURE_TEST_CASE(test_structure, fixture) {
  for (std::size_t i = 0; i < nvars; ++i) {
    std::vector<domain> args1;
    args1.push_back({x[i]});
    if (i > 0) {
      args1.push_back({x[i-1], x[i]});
    }
    if (i < nvars - 1) {
      args1.push_back({x[i], x[i+1]});
    }

    std::vector<domain> args2;
    for (std::size_t id : fg.neighbors(x[i])) {
      args2.push_back(fg.arguments(id));
    }

    boost::sort(args1, domain_less());
    boost::sort(args2, domain_less());
    BOOST_CHECK_EQUAL(args1, args2);
  }
  BOOST_CHECK_EQUAL(fg.num_arguments(), nvars);
  BOOST_CHECK_EQUAL(fg.num_factors(), 2*nvars - 1);
  BOOST_CHECK_EQUAL(fg.num_edges(), 2*(nvars-1) + nvars);
}

BOOST_FIXTURE_TEST_CASE(test_simplify, fixture) {
  fg.simplify();
  for (std::size_t i = 0; i < nvars; ++i) {
    std::vector<domain> args1;
    if (i > 0) {
      args1.push_back({x[i-1], x[i]});
    }
    if (i < nvars - 1) {
      args1.push_back({x[i], x[i+1]});
    }

    std::vector<domain> args2;
    for (std::size_t id : fg.neighbors(x[i])) {
      args2.push_back(fg.arguments(id));
    }

    boost::sort(args1, domain_less());
    boost::sort(args2, domain_less());
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
