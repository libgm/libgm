#define BOOST_TEST_MODULE hybrid_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/hybrid_dataset_io.hpp>
#include <libgm/math/constants.hpp>

#include "../../math/eigen/helpers.hpp"

#include <random>

namespace libgm {
  template class hybrid_dataset<double>;
  template class hybrid_dataset<float>;
}

using namespace libgm;

typedef std::pair<hybrid_index<double>, double> sample_type;
typedef std::pair<hybrid_assignment<>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_index);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  domain fv = u.new_finite_variables(3, "fv", 3);
  domain vv = u.new_vector_variables(1, "vv", 2);
  hybrid_domain<> v(fv, vv);
  
  hybrid_dataset<> ds(v);
  BOOST_CHECK(ds.empty());
  
  // insert a sample
  hybrid_index<double> values;
  values.finite() = {2, 0, 1};
  values.vector() = vec2(2.0, 0.0);
  ds.insert(values, 0.5);

  // insert an assignment
  hybrid_assignment<> a;
  a.finite()[fv[0]] = 1;
  a.finite()[fv[1]] = 2;
  a.finite()[fv[2]] = 0;
  a.vector()[vv[0]] = vec2(0.0, 0.5);
  ds.insert(a, 0.7);

  // insert a bunch of empty samples
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  size_t i = 0;
  for (const auto& s : ds) {
    std::cout << i << ": " << s.first << " " << s.second << std::endl;
    ++i;
  }

  // check the number of samples
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  hybrid_dataset<>::const_iterator it, end;
  const auto& cds = ds;
  std::tie(it, end) = cds(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.finite(), finite_index({2, 0, 1}));
  BOOST_CHECK_EQUAL(it->first.vector(), vec2(2.0, 0.0));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds[0]);
  BOOST_CHECK_EQUAL(*it, ds(0, v));
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.finite(), finite_index({1, 2, 0}));
  BOOST_CHECK_EQUAL(it->first.vector(), vec2(0.0, 0.5));
  BOOST_CHECK_EQUAL(it->second, 0.7);
  BOOST_CHECK_EQUAL(*it, ds[1]);
  BOOST_CHECK_EQUAL(*it, ds(1, v));
  ++it;

  // check the remaining samples
  std::vector<size_t> rest(3, -1);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.finite(), rest);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    for (size_t j = 0; j < 2; ++j) {
      BOOST_CHECK(std::isnan(it->first.vector()[j]));
    }
    ++it;
  }
  
  // check that we covered all the samples
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  domain fv = u.new_finite_variables(3, "fv", 3);
  domain vv = u.new_vector_variables(3, "vv", 2);
  hybrid_domain<> v(fv, vv);
  
  hybrid_dataset<> ds(v);
  ds.insert(1);

  hybrid_dataset<>::iterator it1, end1;
  std::tie(it1, end1) = ds(v);

  hybrid_dataset<>::const_iterator it2 = ds.begin();
  hybrid_dataset<>::const_iterator end2 = ds.end();

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
  domain fv = u.new_finite_variables(3, "fv", 3);
  domain vv = u.new_vector_variables(1, "vv", 2);
  hybrid_domain<> v(fv, vv);
  
  hybrid_dataset<> ds(v);

  // insert 2 samples
  hybrid_index<double> values;
  values.finite() = finite_index{2, 0, 1};
  values.vector() = vec2(2.0, 1.5);
  ds.insert(values, 0.5);
  values.finite() = finite_index{1, 1, 2};
  values.vector() = vec2(1.0, 0.5);
  ds.insert(values, 0.2);

  hybrid_dataset<>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(v);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.finite().size(), 3);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[0]), 2);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[1]), 0);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[2]), 1);
  BOOST_CHECK_EQUAL(it->first.vector().size(), 1);
  BOOST_CHECK_EQUAL(it->first.vector().at(vv[0]), vec2(2.0, 1.5));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, v));
  BOOST_CHECK(!it.end());
  ++it;
  
  // check the second sample
  BOOST_CHECK_EQUAL(it->first.finite().size(), 3);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[0]), 1);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[1]), 1);
  BOOST_CHECK_EQUAL(it->first.finite().at(fv[2]), 2);
  BOOST_CHECK_EQUAL(it->first.vector().size(), 1);
  BOOST_CHECK_EQUAL(it->first.vector().at(vv[0]), vec2(1.0, 0.5));
  BOOST_CHECK_EQUAL(it->second, 0.2);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, v));
  BOOST_CHECK(!it.end());
  ++it;

  // check if finisehd
  BOOST_CHECK(it == end);
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_sample) {
  universe u;
  domain fv = u.new_finite_variables(1, "fv", 2);
  domain vv = u.new_vector_variables(1, "vv", 1);
  hybrid_domain<> v(fv, vv);

  hybrid_dataset<> ds(v);

  // insert two samples, one (0, 1), and the other (1, -1)
  hybrid_index<double> v0(finite_index({0}), vec1(1));
  hybrid_index<double> v1(finite_index({2}), vec1(-1));
  ds.insert(v0, 1.0);
  ds.insert(v1, 1.0);

  // check that the sample average is approximately (0.5, 0.0)
  std::uniform_int_distribution<size_t> unif(0, ds.size() - 1);
  size_t nsamples = 500;
  double fsum = 0.0;
  double vsum = 0.0;
  std::mt19937 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    auto s = ds[unif(rng)];
    fsum += s.first.finite()[0] * s.second;
    vsum += s.first.vector()[0] * s.second;
  }

  BOOST_CHECK_SMALL(fsum / nsamples - 1.0, 0.05);
  BOOST_CHECK_SMALL(vsum / nsamples, 0.05);
}

BOOST_AUTO_TEST_CASE(test_shuffle) {
  universe u;
  domain fv = u.new_finite_variables(1, "fv", 3);
  domain vv = u.new_vector_variables(1, "vv", 1);
  hybrid_domain<> v(fv, vv);

  hybrid_dataset<> ds(v);

  // insert two samples, one (0, 1), and the other (1, -1)
  sample_type s0(hybrid_index<double>(finite_index({0}), vec1(1)), 0.2);
  sample_type s1(hybrid_index<double>(finite_index({2}), vec1(-1)), 0.5);
  ds.insert(s0);
  ds.insert(s1);

  // repeatedly shuffle and check if both permutations have the same frequency
  std::mt19937 rng;
  int norig = 0;
  int nswap = 0;
  int nbad = 0;
  size_t nshuffles = 500;
  for (size_t i = 0; i < nshuffles; ++i) {
    ds.shuffle(rng);
    if (ds.size() == 2 && ds.arguments() == v) {
      if (ds[0] == s0 && ds[1] == s1) {
        ++norig;
      } else if (ds[0] == s1 && ds[1] == s0) {
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
  symbolic_format format;
  hybrid_dataset<> ds;
  format.load(dir + "/hybrid.cfg", u);
  load(dir + "/hybrid.txt", format, ds);

  size_t fvalues[][2] = { {size_t(-1), 2}, {1, 3}, {0, 0} };
  double vvalues[][2] = { {33.0, nan<double>()}, {22.0, 178.0}, {11.0, 150.0} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  size_t i = 0;
  for (const auto& s : ds) {
    BOOST_CHECK_EQUAL(s.first.finite()[0], fvalues[i][0]);
    BOOST_CHECK_EQUAL(s.first.finite()[1], fvalues[i][1]);
    BOOST_CHECK_CLOSE(s.first.vector()[0], vvalues[i][0], 1e-10);
    if (std::isnan(vvalues[i][1])) {
      BOOST_CHECK(std::isnan(s.first.vector()[1]));
    } else {
      BOOST_CHECK_CLOSE(s.first.vector()[1], vvalues[i][1], 1e-10);
    }
    BOOST_CHECK_EQUAL(s.second, weights[i]);
    ++i;
  }

  save("hybrid.tmp", format, ds);
}
