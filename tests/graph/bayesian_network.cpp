#define BOOST_TEST_MODULE bayesian_network
#include <boost/test/unit_test.hpp>

#include <libgm/model/bayesian_network.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/canonical_gaussian.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/utility/operations.hpp>
#include <libgm/learning/dataset/uint_dataset.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

#include "predicates.hpp"

using namespace libgm;

typedef canonical_gaussian<var> cgaussian;
typedef moment_gaussian<var> mgaussian;
typedef probability_table<var> ptable;

namespace libgm {
  template class bayesian_network<ptable>;
  template class bayesian_network<cgaussian>;
  template class bayesian_network<mgaussian>;
}

struct fixture {
  fixture() {
    for (std::size_t i = 0; i < 5; ++i) {
      x.push_back(var::discrete(u, "x" + std::to_string(i), 2));
    }

    /* Create factors for a Bayesian network with this structure:
     * 0, 1 (no parents)
     * 1 --> 2
     * 1,2 --> 3
     * 0,3 --> 4
     */

    f0   = ptable({x[0]}, {0.3, 0.7});
    f1   = ptable({x[1]}, {0.5, 0.5});
    f21  = ptable({x[2], x[1]}, {0.8, 0.2, 0.2, 0.8});
    f312 = ptable({x[3], x[1], x[2]}, {0.1, 0.9, 0.1, 0.9, 0.3, 0.7, 0.5, 0.5});
    f403 = ptable({x[4], x[0], x[3]}, {0.6, 0.4, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9});

    bn.add_factor(x[0], f0);
    bn.add_factor(x[1], f1);
    bn.add_factor(x[2], f21);
    bn.add_factor(x[3], f312);
    bn.add_factor(x[4], f403);
  }

  universe u;
  domain<var> x;
  ptable f0, f1, f21, f312, f403;
  bayesian_network<ptable> bn;
};

/*
BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(bn, u));
}
*/

BOOST_FIXTURE_TEST_CASE(test_markov_graph, fixture) {
  typedef std::pair<var, var> vpair;
  std::vector<vpair> vpairs =
    {vpair(x[1], x[2]), vpair(x[1], x[3]), vpair(x[2], x[3]),
     vpair(x[0], x[3]), vpair(x[0], x[4]), vpair(x[3], x[4])};
  undirected_graph<var> mg(vpairs);
  undirected_graph<var> mg2;
  bn.markov_graph(mg2);
  BOOST_CHECK_EQUAL(mg, mg2);
}

BOOST_FIXTURE_TEST_CASE(test_conditioning, fixture) {
  uint_assignment<var> a;
  a[x[0]] = 0;
  a[x[1]] = 1;
  double likelihood(bn.condition(a));
  std::vector<ptable> factors = {f0, f1, f21, f312, f403};
  ptable marginal = prod_all(factors).marginal({x[0], x[1]});
  BOOST_CHECK_CLOSE(likelihood, marginal(a), 1e-5);

  bayesian_network<ptable> bn2;
  bn2.add_factor(x[2], f21.restrict(a));
  bn2.add_factor(x[3], f312.restrict(a));
  bn2.add_factor(x[4], f403.restrict(a));
  BOOST_CHECK(model_close_log_likelihoods(bn, bn2, 1e-6));
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  uint_dataset<var> ds(x);
  uint_assignment<var> a;
  std::size_t nsamples = 5000;
  std::mt19937 rng;
  for (std::size_t i = 0; i < nsamples; ++i) {
    bn.sample(rng, a);
    ds.insert(a.values(ds.arguments()), 1.0);
  }

  factor_mle<ptable> mle;
  ptable g0 = mle(ds, {x[0]});
  ptable g1 = mle(ds, {x[1]});
  ptable g21 = mle(ds, {x[2]}, {x[1]});
  ptable g312 = mle(ds, {x[3]}, {x[1], x[2]});
  ptable g403 = mle(ds, {x[4]}, {x[0], x[3]});

  BOOST_CHECK_SMALL(kl_divergence(f0, g0), 1e-2);
  BOOST_CHECK_SMALL(kl_divergence(f1, g1), 1e-2);
  BOOST_CHECK_SMALL(kl_divergence(f21, g21), 1e-2);
  BOOST_CHECK_SMALL(kl_divergence(f312, g312), 1e-2);
  BOOST_CHECK_SMALL(kl_divergence(f403, g403), 1e-2);
}
