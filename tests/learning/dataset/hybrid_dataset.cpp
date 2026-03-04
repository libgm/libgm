#define BOOST_TEST_MODULE hybrid_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/hybrid_dataset_io.hpp>

#include <libgm/argument/hybrid_assignment.hpp>
#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/math/constants.hpp>

#include "../../math/eigen/helpers.hpp"

#include <random>

namespace libgm {
  template class hybrid_dataset<var>;
  template class hybrid_dataset<vec>;
}

using namespace libgm;

typedef std::pair<hybrid_vector<>, double> sample_type;
// typedef std::pair<hybrid_assignment<>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
// BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  domain<vec> disc = { vec::discrete(u, "a", {3,3}), vec::discrete(u, "b", 4) };
  domain<vec> cont = { vec::continuous(u, "x", 2) };
  hybrid_domain<vec> v(disc, cont);

  hybrid_dataset<vec> ds(v);
  BOOST_CHECK(ds.empty());

  // insert a sample
  hybrid_vector<> values;
  values.uint() = {2, 0, 1};
  values.real() = vec2(2.0, 0.0);
  ds.insert(values, 0.5);

  // insert an assignment
  hybrid_assignment<vec> a;
  a.uint()[disc[0]] = {1, 2};
  a.uint()[disc[1]] = {0};
  a.real()[cont[0]] = vec2(0.0, 0.5);
  ds.insert(std::make_pair(a.values(v), 0.7));

  // insert a bunch of empty samples
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  std::size_t i = 0;
  for (const auto& s : ds) {
    std::cout << i << ": " << s.first << " " << s.second << std::endl;
    ++i;
  }

  // check the number of samples
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // check the total weight
  BOOST_CHECK_CLOSE(ds.weight(), 11.2, 1e-6);

  hybrid_dataset<vec>::const_iterator it, end;
  const auto& cds = ds;
  std::tie(it, end) = cds.samples(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.uint(), uint_vector({2, 0, 1}));
  BOOST_CHECK_EQUAL(it->first.real(), vec2(2.0, 0.0));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0));
  BOOST_CHECK_EQUAL(*it, ds.sample(0, v));
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.uint(), uint_vector({1, 2, 0}));
  BOOST_CHECK_EQUAL(it->first.real(), vec2(0.0, 0.5));
  BOOST_CHECK_EQUAL(it->second, 0.7);
  BOOST_CHECK_EQUAL(*it, ds.sample(1));
  BOOST_CHECK_EQUAL(*it, ds.sample(1, v));
  ++it;

  // check the remaining samples
  uint_vector rest(3, -1);
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.uint(), rest);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    for (std::size_t j = 0; j < 2; ++j) {
      BOOST_CHECK(std::isnan(it->first.real()[j]));
    }
    ++it;
  }

  // check that we covered all the samples
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  hybrid_domain<vec> v;
  for (std::size_t i = 0; i < 3; ++i) {
    v.push_back(vec::discrete(u, "a", 3));
    v.push_back(vec::continuous(u, "x", 2));
  }

  hybrid_dataset<vec> ds(v);
  ds.insert(1);

  hybrid_dataset<vec>::iterator it1, end1;
  std::tie(it1, end1) = ds.samples(v);

  hybrid_dataset<vec>::const_iterator it2 = ds.begin();
  hybrid_dataset<vec>::const_iterator end2 = ds.end();

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
  domain<var> disc = { var::discrete(u, "a", 3), var::discrete(u, "b", 3),
                       var::discrete(u, "c", 3) };
  domain<var> cont = { var::continuous(u, "x"), var::continuous(u, "y") };
  hybrid_domain<var> v(disc, cont);
  hybrid_dataset<var> ds(v);

  // insert 2 samples
  hybrid_vector<> values;
  values.uint() = uint_vector{2, 0, 1};
  values.real() = vec2(2.0, 1.5);
  ds.insert(values, 0.5);
  values.uint() = uint_vector{1, 1, 2};
  values.real() = vec2(1.0, 0.5);
  ds.insert(values, 0.2);

  hybrid_dataset<var>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.uint().size(), 3);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[0]), 2);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[1]), 0);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[2]), 1);
  BOOST_CHECK_EQUAL(it->first.real().size(), 2);
  BOOST_CHECK_EQUAL(it->first.real().at(cont[0]), 2.0);
  BOOST_CHECK_EQUAL(it->first.real().at(cont[1]), 1.5);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, v));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.uint().size(), 3);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[0]), 1);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[1]), 1);
  BOOST_CHECK_EQUAL(it->first.uint().at(disc[2]), 2);
  BOOST_CHECK_EQUAL(it->first.real().size(), 2);
  BOOST_CHECK_EQUAL(it->first.real().at(cont[0]), 1.0);
  BOOST_CHECK_EQUAL(it->first.real().at(cont[1]), 0.5);
  BOOST_CHECK_EQUAL(it->second, 0.2);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, v));
  BOOST_CHECK(it);
  ++it;

  // check if finisehd
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}
*/

BOOST_AUTO_TEST_CASE(test_weight_iterator) {
  hybrid_dataset<var> ds(hybrid_domain<var>{});

  // insert 2 samples
  hybrid_vector<> values;
  ds.insert(values, 0.5);
  ds.insert(values, 0.2);

  hybrid_dataset<var>::weight_iterator it, end;
  std::tie(it, end) = ds.weights();

  BOOST_CHECK_EQUAL(*it, 0.5);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(*it, 0.2);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_sample) {
  universe u;
  hybrid_domain<var> v;
  v.push_back(var::discrete(u, "a", 2));
  v.push_back(var::continuous(u, "x"));
  hybrid_dataset<var> ds(v);

  // insert two samples, one (0, 1), and the other (1, -1)
  hybrid_vector<> v0(uint_vector({0}), vec1(1));
  hybrid_vector<> v1(uint_vector({2}), vec1(-1));
  ds.insert(v0, 1.0);
  ds.insert(v1, 1.0);

  // check that the sample average is approximately (0.5, 0.0)
  std::uniform_int_distribution<std::size_t> unif(0, ds.size() - 1);
  std::size_t nsamples = 500;
  double uint_sum = 0.0;
  double real_sum = 0.0;
  std::mt19937 rng;
  for (std::size_t i = 0; i < nsamples; ++i) {
    auto s = ds.sample(unif(rng));
    uint_sum += s.first.uint()[0] * s.second;
    real_sum += s.first.real()[0] * s.second;
  }

  BOOST_CHECK_SMALL(uint_sum / nsamples - 1.0, 0.05);
  BOOST_CHECK_SMALL(real_sum / nsamples, 0.05);
}

BOOST_AUTO_TEST_CASE(test_shuffle) {
  universe u;
  hybrid_domain<var> v;
  v.push_back(var::discrete(u, "a", 3));
  v.push_back(var::continuous(u, "x"));
  hybrid_dataset<var> ds(v);

  // insert two samples, one (0, 1), and the other (1, -1)
  sample_type s0({{0}, {1}}, 0.2);
  sample_type s1({{2}, {-1}}, 0.5);
  ds.insert(s0);
  ds.insert(s1);

  // repeatedly shuffle and check if both permutations have the same frequency
  std::mt19937 rng;
  int norig = 0;
  int nswap = 0;
  int nbad = 0;
  std::size_t nshuffles = 500;
  for (std::size_t i = 0; i < nshuffles; ++i) {
    ds.shuffle(rng);
    if (ds.size() == 2 && ds.arguments() == v) {
      if (ds.sample(0) == s0 && ds.sample(1) == s1) {
        ++norig;
      } else if (ds.sample(0) == s1 && ds.sample(1) == s0) {
        ++nswap;
      } else {
        ++nbad;
      }
    } else {
      ++nbad;
    }
  }

  BOOST_CHECK_EQUAL(nbad, 0);
  BOOST_CHECK_SMALL(double(norig - nswap) / nshuffles, 0.05);
}

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  universe u;
  text_dataset_format<var> format;
  hybrid_dataset<var> ds;
  format.load_config(dir + "/hybrid.cfg", u);
  load(dir + "/hybrid.txt", format, ds);

  std::size_t uint_values[][2] = { {std::size_t(-1), 2}, {1, 3}, {0, 0} };
  double real_values[][2] = { {33, nan<double>()}, {22, 178}, {11, 150} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  std::size_t i = 0;
  for (const auto& s : ds) {
    BOOST_CHECK_EQUAL(s.first.uint()[0], uint_values[i][0]);
    BOOST_CHECK_EQUAL(s.first.uint()[1], uint_values[i][1]);
    BOOST_CHECK_CLOSE(s.first.real()[0], real_values[i][0], 1e-10);
    if (std::isnan(real_values[i][1])) {
      BOOST_CHECK(std::isnan(s.first.real()[1]));
    } else {
      BOOST_CHECK_CLOSE(s.first.real()[1], real_values[i][1], 1e-10);
    }
    BOOST_CHECK_EQUAL(s.second, weights[i]);
    ++i;
  }

  save("hybrid.tmp", format, ds);
}
