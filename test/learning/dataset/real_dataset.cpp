#define BOOST_TEST_MODULE real_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/moment_gaussian.hpp>
#include <libgm/learning/dataset/real_dataset.hpp>
#include <libgm/learning/dataset/real_dataset_io.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>
#include <libgm/math/constants.hpp>

#include "../../math/eigen/helpers.hpp"

#include <random>

namespace libgm {
  template class basic_dataset<real_data_traits<double, variable> >;
  template class basic_dataset<real_data_traits<float, variable> >;
}

using namespace libgm;

typedef real_vector<double> vec_type;
typedef std::pair<vec_type, double> sample_type;
typedef std::pair<real_assignment<>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  domain v = u.new_continuous_variables(2, "v", 1);
  v.push_back(u.new_continuous_variable("w", 2));

  real_dataset<> ds;
  ds.initialize(v);
  BOOST_CHECK(ds.empty());

  // insert a record
  vec_type values(4);
  values[0] = 2.0;
  values[1] = 0.0;
  values[2] = 1.0;
  values[3] = 1.5;
  ds.insert(values, 0.5);

  // insert a vector assignment
  real_assignment<> a;
  a[v[0]] = vec1(1.0);
  a[v[1]] = vec1(2.0);
  a[v[2]] = vec2(0.0, 0.5);
  ds.insert(a, 0.7);

  // insert a bunch of empty records
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  std::size_t i = 0;
  for (const auto& s : ds) {
    std::cout << i << ": " << s.first.transpose() << "\t" << s.second
              << std::endl;
    ++i;
  }

  // check the number of samples
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // check the total weight
  BOOST_CHECK_CLOSE(ds.weight(), 11.2, 1e-6);

  // value iteraotr checks
  real_dataset<>::const_iterator it, end;
  const auto& cds = ds;
  std::tie(it, end) = cds(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first, vec4(2.0, 0.0, 1.0, 1.5));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds[0]);
  BOOST_CHECK_EQUAL(*it, ds(0, v));
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first, vec4(1.0, 2.0, 0.0, 0.5));
  BOOST_CHECK_EQUAL(it->second, 0.7);
  BOOST_CHECK_EQUAL(*it, ds[1]);
  BOOST_CHECK_EQUAL(*it, ds(1, v));
  ++it;

  // check the remaining samples
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.size(), 4);
    for (std::size_t j = 0; j < 4; ++j) {
      BOOST_CHECK(std::isnan(it->first[j]));
    }
    BOOST_CHECK_EQUAL(it->second, 1.0);
    BOOST_CHECK_EQUAL(it->second, ds[i+2].second);
    ++it;
  }

  // check that we covered all the samples
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  domain v = u.new_continuous_variables(3, "v", 1);

  real_dataset<> ds;
  ds.initialize(v);
  ds.insert(1);

  real_dataset<>::iterator it1, end1;
  std::tie(it1, end1) = ds(v);

  real_dataset<>::const_iterator it2 = ds.begin();
  real_dataset<>::const_iterator end2 = ds.end();

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


BOOST_AUTO_TEST_CASE(test_assignment_iterator) {
  universe u;
  variable x = u.new_continuous_variable("x", 1);
  variable y = u.new_continuous_variable("y", 1);
  variable z = u.new_continuous_variable("z", 2);
  domain v = {x, y, z};

  real_dataset<> ds(v);

  // insert 2 records
  ds.insert(vec4(2.0, 0.0, 1.0, 0.8), 0.5);
  ds.insert(vec4(1.0, 1.0, 2.0, 0.4), 0.2);

  real_dataset<>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(x), vec1(2.0));
  BOOST_CHECK_EQUAL(it->first.at(y), vec1(0.0));
  BOOST_CHECK_EQUAL(it->first.at(z), vec2(1.0, 0.8));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, v));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(x), vec1(1.0));
  BOOST_CHECK_EQUAL(it->first.at(y), vec1(1.0));
  BOOST_CHECK_EQUAL(it->first.at(z), vec2(2.0, 0.4));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, v));
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

BOOST_AUTO_TEST_CASE(test_weight_iterator) {
  domain v;
  real_dataset<> ds(v);

  // insert 2 records
  ds.insert(real_vector<>(), 0.5);
  ds.insert(real_vector<>(), 0.2);

  real_dataset<>::weight_iterator it, end;
  std::tie(it, end) = ds.weights();

  // check the first sample
  BOOST_CHECK_EQUAL(*it, 0.5);
  BOOST_CHECK(it != end);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(*it, 0.2);
  BOOST_CHECK(it != end);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
}

struct fixture {
  universe u;
  domain v;
  real_dataset<> ds;
  mgaussian f;
  factor_mle<mgaussian> mle;
  std::mt19937 rng;

  fixture()  {
    v = u.new_continuous_variables(3, "v", 1);
    ds.initialize(v, 1000);
    f = mgaussian(v, vec3(0.5, 1, 2), mat33(3, 2, 1, 2, 2, 1, 1, 1, 2));
    auto d = f.distribution();
    for (std::size_t i = 0; i < 1000; ++i) {
      ds.insert(d(rng), 1.0);
    }
  }
};

BOOST_FIXTURE_TEST_CASE(test_reconstruction, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  for (std::size_t i = 0; i < num_dimensions(v); ++i) {
    for (std::size_t j = i; j < num_dimensions(v); ++j) {
      domain dom = domain({v[i], v[j]}).unique();
      double kl = kl_divergence(f.marginal(dom), mle(ds, dom));
      std::cout << dom << ": " << kl << std::endl;
      BOOST_CHECK_SMALL(kl, 1e-2);
    }
  }

  // fill the content of the dataset using mutable iteration
  domain v01 = {v[0], v[1]};
  for (auto& sample : ds(v01)) {
    sample.first[0] = nan<double>();
    sample.first[1] = nan<double>();
  }

  // verify that we get the mutated version back
  for (const auto& s : ds(v01)) {
    BOOST_CHECK_EQUAL(s.first.size(), 2);
    BOOST_CHECK(std::isnan(s.first[0]));
    BOOST_CHECK(std::isnan(s.first[1]));
    BOOST_CHECK_EQUAL(s.second, 1.0);
  }

  // verify that the marginal over v[2] is still good
  domain dom2 = {v[2]};
  double kl = kl_divergence(f.marginal(dom2), mle(ds, dom2));
  std::cout << "Rest: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  // draw samples from the dataset and attempt to recover the mean
  vec_type mean = vec_type::Zero(3);
  std::uniform_int_distribution<std::size_t> unif(0, ds.size() - 1);
  for (std::size_t i = 0; i < 600; ++i) {
    auto sample = ds[unif(rng)];
    mean += sample.first * sample.second;
  }
  mean /= 600;

  double diff = (mean - f.mean()).norm();
  std::cout << "Samples: " << diff << std::endl;
  BOOST_CHECK_SMALL(diff, 0.2);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  mgaussian mle1 = mle(ds, v);
  ds.shuffle(rng);
  mgaussian mle2 = mle(ds, v);
  double kl = kl_divergence(mle1, mle2);
  std::cout << "Shuffle: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-10);
}

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  universe u;
  text_dataset_format format;
  real_dataset<> ds;
  format.load_config(dir + "/vector.cfg", u);
  load(dir + "/vector.txt", format, ds);

  double values[][3] = { {180, 0, 0}, {178.2, 1, 0}, {nan<double>(), 2, 2} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  std::size_t i = 0;
  for (const auto& s : ds) {
    if (std::isnan(values[i][0])) {
      BOOST_CHECK(std::isnan(s.first[0]));
    } else {
      BOOST_CHECK_CLOSE(s.first[0], values[i][0], 1e-10);
    }
    BOOST_CHECK_CLOSE(s.first[1], values[i][1], 1e-10);
    BOOST_CHECK_CLOSE(s.first[2], values[i][2], 1e-10);
    BOOST_CHECK_EQUAL(s.second, weights[i]);
    ++i;
  }

  save("vector.tmp", format, ds);
}
