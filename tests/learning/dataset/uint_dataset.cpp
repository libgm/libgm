#define BOOST_TEST_MODULE uint_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/uint_dataset.hpp>
#include <libgm/learning/dataset/uint_dataset_io.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

namespace libgm {
  template class basic_dataset<var, uint_vector, double>;
  template class basic_dataset<vec, uint_vector, double>;
}

using namespace libgm;

typedef std::pair<uint_vector, double> sample_type;
// typedef std::pair<uint_assignment<var>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
// BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

domain<var> new_discrete(universe& u, std::size_t count, std::size_t values) {
  domain<var> dom;
  for (std::size_t i = 0; i < count; ++i) {
    dom.push_back(var::discrete(u, "v" + std::to_string(i), values));
  }
  return dom;
}

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  domain<var> v = new_discrete(u, 3, 3);

  uint_dataset<var> ds;
  ds.initialize(v);
  BOOST_CHECK(ds.empty());

  // insert a sample
  uint_vector values(3);
  values[0] = 2;
  values[1] = 0;
  values[2] = 1;
  ds.insert(values, 0.5);

  // insert a pair
  values[0] = 1;
  values[1] = 2;
  values[2] = 0;
  ds.insert(std::make_pair(values, 0.7));

  // insert a bunch of empty samples
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  std::size_t i = 0;
  for (const auto& sample : ds) {
    std::cout << i << " ";
    for (std::size_t x : sample.first) { std::cout << x << " "; }
    std::cout << sample.second << std::endl;
    ++i;
  }

  // check the number of samples
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // check the weight
  BOOST_CHECK_CLOSE(ds.weight(), 11.2, 1e-6);

  // value iterator checks
  uint_dataset<var>::const_iterator it, end;
  const auto& cds = ds;
  std::tie(it, end) = cds.samples(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first, uint_vector({2, 0, 1}));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0));
  BOOST_CHECK_EQUAL(*it, ds.sample(0, v));
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first, uint_vector({1, 2, 0}));
  BOOST_CHECK_EQUAL(it->second, 0.7);
  BOOST_CHECK_EQUAL(*it, ds.sample(1));
  BOOST_CHECK_EQUAL(*it, ds.sample(1, v));
  ++it;

  // check the remaining samples
  uint_vector rest(3, -1);
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first, rest);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    BOOST_CHECK_EQUAL(*it, ds.sample(i+2));
    BOOST_CHECK_EQUAL(*it, ds.sample(i+2, v));
    ++it;
  }

  // check that we covered all the samples
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  domain<var> v = new_discrete(u, 3, 3);

  uint_dataset<var> ds(v);
  ds.insert(1);

  uint_dataset<var>::iterator it1, end1;
  std::tie(it1, end1) = ds.samples(v);

  uint_dataset<var>::const_iterator it2 = ds.begin();
  uint_dataset<var>::const_iterator end2 = ds.end();

  BOOST_CHECK(it1 == it2);
  BOOST_CHECK(it2 == it1);

  BOOST_CHECK(end1 == end2);
  BOOST_CHECK(end2 == end1);

  BOOST_CHECK(it1 != end1);
  BOOST_CHECK(it1 != end2);
  BOOST_CHECK(it2 != end1);
  BOOST_CHECK(it2 != end2);
  BOOST_CHECK(it1);
  BOOST_CHECK(it2);

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
  BOOST_CHECK(!it1);
  BOOST_CHECK(!it2);
}

/*
BOOST_AUTO_TEST_CASE(test_assignment_iterator) {
  universe u;
  domain<var> v = new_discrete(3, 3);

  uint_dataset<var> ds(v);

  // insert 2 samples
  ds.insert(uint_vector{2, 0, 1}, 0.5);
  ds.insert(uint_vector{1, 1, 2}, 0.2);

  uint_dataset<var>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(v[0]), 2);
  BOOST_CHECK_EQUAL(it->first.at(v[1]), 0);
  BOOST_CHECK_EQUAL(it->first.at(v[2]), 1);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, v));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(v[0]), 1);
  BOOST_CHECK_EQUAL(it->first.at(v[1]), 1);
  BOOST_CHECK_EQUAL(it->first.at(v[2]), 2);
  BOOST_CHECK_EQUAL(it->second, 0.2);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, v));
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}
*/

BOOST_AUTO_TEST_CASE(test_weight_iterator) {
  universe u;
  domain<var> v = new_discrete(u, 3, 3);

  uint_dataset<var> ds(v);
  ds.insert(uint_vector{2, 0, 1}, 0.5);
  ds.insert(uint_vector{1, 1, 2}, 0.2);

  uint_dataset<var>::weight_iterator it, end;
  std::tie(it, end) = ds.weights();

  BOOST_CHECK_EQUAL(*it, 0.5);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(*it, 0.2);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it == end);
}

struct fixture {
  typedef probability_table<var> ptable;

  universe u;
  domain<var> v;
  uint_dataset<var> ds;
  ptable f;
  factor_mle<ptable> mle;
  std::mt19937 rng;

  fixture() {
    v = new_discrete(u, 3, 2);
    ds.initialize(v, 1000);

    f = uniform_table_generator<ptable>()(v, rng).normalize();
    auto d = f.distribution();
    for (std::size_t i = 0; i < 1000; ++i) {
      ds.insert(d(rng), 1.0);
    }
  }
};

BOOST_FIXTURE_TEST_CASE(test_reconstruction, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  for (std::size_t i = 0; i < v.size(); ++i) {
    for (std::size_t j = i; j < v.size(); ++j) {
      domain<var> dom = { v[i], v[j] };
      dom.unique();
      double kl = kl_divergence(f.marginal(dom), mle(ds, dom));
      std::cout << dom << ": " << kl << std::endl;
      BOOST_CHECK_SMALL(kl, 1e-2);
    }
  }

  // fill the content of the dataset using mutable iteration
  domain<var> v01 = {v[0], v[1]};
  for (auto& sample : ds.samples(v01)) {
    sample.first[0] = 1;
    sample.first[1] = 0;
  }

  // verify that we get the mutated version back
  for (const auto& sample : ds.samples(v01)) {
    BOOST_CHECK_EQUAL(sample.first.size(), 2);
    BOOST_CHECK_EQUAL(sample.first[0], 1);
    BOOST_CHECK_EQUAL(sample.first[1], 0);
  }

  // verify that the marginal over v[2] is still good
  domain<var> dom2 = {v[2]};
  double kl = kl_divergence(f.marginal(dom2), mle(ds, dom2));
  std::cout << "Rest: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  // draw samples from the dataset and attempt to recover f
  ptable g(v, 0.0);
  std::uniform_int_distribution<std::size_t> unif(0, ds.size() - 1);
  for (std::size_t i = 0; i < 500; ++i) {
    auto sample = ds.sample(unif(rng));
    g.param(sample.first) += sample.second;
  }
  g.normalize();
  double kl = kl_divergence(f, g);
  std::cout << "Samples: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 0.05);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  ptable mle1 = mle(ds, v);
  ds.shuffle(rng);
  ptable mle2 = mle(ds, v);
  double kl = kl_divergence(mle1, mle2);
  std::cout << "Shuffle: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-10);
}

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  universe u;
  text_dataset_format<var> format;
  uint_dataset<var> ds;
  format.load_config(dir + "/uint.cfg", u);
  load(dir + "/uint.txt", format, ds);

  std::size_t values[][3] = { {0, std::size_t(-1), 2}, {2, 1, 3}, {1, 0, 0} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  std::size_t i = 0;
  for (const auto& sample : ds) {
    BOOST_CHECK_EQUAL(sample.first[0], values[i][0]);
    BOOST_CHECK_EQUAL(sample.first[1], values[i][1]);
    BOOST_CHECK_EQUAL(sample.first[2], values[i][2]);
    BOOST_CHECK_EQUAL(sample.second, weights[i]);
    ++i;
  }

  save("uint.tmp", format, ds);
}
