#define BOOST_TEST_MODULE finite_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/learning/dataset/finite_dataset.hpp>
#include <libgm/learning/dataset/finite_dataset_io.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>

namespace libgm {
  template class basic_dataset<finite_data_traits<double, variable> >;
  template class basic_dataset<finite_data_traits<float, variable> >;
}

using namespace libgm;

typedef std::pair<finite_index, double> sample_type;
typedef std::pair<finite_assignment<>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_index);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  domain v = u.new_finite_variables(3, "v", 3);
  
  finite_dataset<> ds;
  ds.initialize(v);
  BOOST_CHECK(ds.empty());
  
  // insert a sample
  finite_index values(3);
  values[0] = 2;
  values[1] = 0;
  values[2] = 1;
  ds.insert(values, 0.5);

  // insert a finite assignment
  finite_assignment<> a;
  a[v[0]] = 1;
  a[v[1]] = 2;
  a[v[2]] = 0;
  ds.insert(a, 0.7);

  // insert a bunch of empty samples
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  size_t i = 0;
  for (const auto& sample : ds) {
    std::cout << i << " ";
    for (size_t x : sample.first) { std::cout << x << " "; }
    std::cout << sample.second << std::endl;
    ++i;
  }

  // check the number of samples
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // value iterator checks
  finite_dataset<>::const_iterator it, end;
  const auto& cds = ds;
  std::tie(it, end) = cds(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first, finite_index({2, 0, 1}));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds[0]);
  BOOST_CHECK_EQUAL(*it, ds(0, v));
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first, finite_index({1, 2, 0}));
  BOOST_CHECK_EQUAL(it->second, 0.7);
  BOOST_CHECK_EQUAL(*it, ds[1]);
  BOOST_CHECK_EQUAL(*it, ds(1, v));
  ++it;

  // check the remaining samples
  std::vector<size_t> rest(3, -1);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first, rest);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    BOOST_CHECK_EQUAL(*it, ds[i+2]);
    BOOST_CHECK_EQUAL(*it, ds(i+2, v));
    ++it;
  }
  
  // check that we covered all the samples
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  domain v = u.new_finite_variables(3, "v", 3);
  
  finite_dataset<> ds;
  ds.initialize(v);
  ds.insert(1);

  finite_dataset<>::iterator it1, end1;
  std::tie(it1, end1) = ds(v);

  finite_dataset<>::const_iterator it2 = ds.begin();
  finite_dataset<>::const_iterator end2 = ds.end();

  BOOST_CHECK(it1 == it2);
  BOOST_CHECK(it2 == it1);

  BOOST_CHECK(end1 == end2);
  BOOST_CHECK(end2 == end1);

  BOOST_CHECK(it1 != end1);
  BOOST_CHECK(it1 != end2);
  BOOST_CHECK(it2 != end1);
  BOOST_CHECK(it2 != end2);
  BOOST_CHECK(!it1.end());
  BOOST_CHECK(!it2.end());

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
  BOOST_CHECK(it1.end());
  BOOST_CHECK(it2.end());
}

BOOST_AUTO_TEST_CASE(test_assignment_iterators) {
  universe u;
  domain v = u.new_finite_variables(3, "v", 3);
  
  finite_dataset<> ds;
  ds.initialize(v);

  // insert 2 samples
  ds.insert(finite_index{2, 0, 1}, 0.5);
  ds.insert(finite_index{1, 1, 2}, 0.2);
  
  finite_dataset<>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(v[0]), 2);
  BOOST_CHECK_EQUAL(it->first.at(v[1]), 0);
  BOOST_CHECK_EQUAL(it->first.at(v[2]), 1);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, v));
  BOOST_CHECK(!it.end());
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(v[0]), 1);
  BOOST_CHECK_EQUAL(it->first.at(v[1]), 1);
  BOOST_CHECK_EQUAL(it->first.at(v[2]), 2);
  BOOST_CHECK_EQUAL(it->second, 0.2);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, v));
  BOOST_CHECK(!it.end());
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(it.end());
}

struct fixture {
  universe u;
  domain v;
  finite_dataset<> ds;
  ptable f;
  factor_mle<ptable> mle;
  std::mt19937 rng;

  fixture() {
    v = u.new_finite_variables(3, "v", 2);
    ds.initialize(v, 1000);
    
    f = uniform_table_generator<ptable>()(v, rng).normalize();
    auto d = f.distribution();
    for (size_t i = 0; i < 1000; ++i) {
      ds.insert(d(rng), 1.0);
    }
  }
};

BOOST_FIXTURE_TEST_CASE(test_reconstruction, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = i; j < v.size(); ++j) {
      domain dom = domain({v[i], v[j]}).unique();
      double kl = kl_divergence(f.marginal(dom), mle(ds, dom));
      std::cout << dom << ": " << kl << std::endl;
      BOOST_CHECK_SMALL(kl, 1e-2);
    }
  }

  // fill the content of the dataset using mutable iteration
  domain v01 = {v[0], v[1]};
  for (auto& sample : ds(v01)) {
    sample.first[0] = 1;
    sample.first[1] = 0;
  }

  // verify that we get the mutated version back
  for (const auto& sample : ds(v01)) {
    BOOST_CHECK_EQUAL(sample.first.size(), 2);
    BOOST_CHECK_EQUAL(sample.first[0], 1);
    BOOST_CHECK_EQUAL(sample.first[1], 0);
  }

  // verify that the marginal over v[2] is still good
  domain dom2 = {v[2]};
  double kl = kl_divergence(f.marginal(dom2), mle(ds, dom2));
  std::cout << "Rest: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  // draw samples from the dataset and attempt to recover f
  ptable g(v, 0.0);
  std::uniform_int_distribution<size_t> unif(0, ds.size() - 1);
  for (size_t i = 0; i < 500; ++i) {
    auto sample = ds[unif(rng)];
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
  symbolic_format format;
  finite_dataset<> ds;
  format.load(dir + "/finite.cfg", u);
  load(dir + "/finite.txt", format, ds);

  size_t values[][3] = { {0, size_t(-1), 2}, {2, 1, 3}, {1, 0, 0} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  size_t i = 0;
  for (const auto& sample : ds) {
    BOOST_CHECK_EQUAL(sample.first[0], values[i][0]);
    BOOST_CHECK_EQUAL(sample.first[1], values[i][1]);
    BOOST_CHECK_EQUAL(sample.first[2], values[i][2]);
    BOOST_CHECK_EQUAL(sample.second, weights[i]);
    ++i;
  }

  save("finite.tmp", format, ds);
}
