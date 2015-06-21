#define BOOST_TEST_MODULE uint_sequence_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset_io.hpp>

#include <random>

namespace libgm {
  template class basic_sequence_dataset<uint_sequence_traits<double, variable> >;
  template class basic_sequence_dataset<uint_sequence_traits<float, variable> >;
}

using namespace libgm;

typedef real_matrix<std::size_t> data_type;
typedef std::pair<data_type, double> sample_type;
typedef std::pair<uint_assignment<>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

struct fixture {
  universe u;
  dprocess_domain p;
  data_type seq0, seq1;
  uint_sequence_dataset<> ds;

  fixture()
    : p(3), seq0(3, 2), seq1(3, 1) {
    p[0] = u.new_discrete_dprocess("a", 2);
    p[1] = u.new_discrete_dprocess("b", 3);
    p[2] = u.new_discrete_dprocess("c", 4);
    seq0 << 0, 1, 1, 2, 2, 3;
    seq1 << 1, 2, 3;
    ds.initialize(p);
    BOOST_CHECK(ds.empty());
    ds.insert(seq0, 0.5);
    ds.insert(seq1, 1.0);
  }
};

BOOST_FIXTURE_TEST_CASE(test_insert, fixture) {
  ds.insert(10);

  // print the datset
  std::cout << ds << std::endl;

  // check the size of the dataset
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // check the total weight
  BOOST_CHECK_CLOSE(ds.weight(), 11.5, 1e-6);

  // direct iteration
  uint_sequence_dataset<>::const_iterator it = ds.begin();
  uint_sequence_dataset<>::const_iterator end = ds.end();
  BOOST_CHECK_EQUAL(it->first, seq0);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds[0]);
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds[1]);
  ++it;
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.rows(), 3);
    BOOST_CHECK_EQUAL(it->first.cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);

  // indirect iteration
  // direct iteration
  dprocess_domain p12 = {p[1], p[2]};
  const auto& cds = ds;
  std::tie(it, end) = cds(p12);
  BOOST_CHECK_EQUAL(it->first, seq0.block(1, 0, 2, 2));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds(0, p12));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds(1, p12));
  ++it;
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.rows(), 2);
    BOOST_CHECK_EQUAL(it->first.cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(test_value_iterators, fixture) {
  uint_sequence_dataset<>::iterator it1, end1;
  std::tie(it1, end1) = ds(p);

  uint_sequence_dataset<>::const_iterator it2 = ds.begin();
  uint_sequence_dataset<>::const_iterator end2 = ds.end();

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

  ++it1;
  ++it2;
  BOOST_CHECK(it1);
  BOOST_CHECK(it2);

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
  BOOST_CHECK(!it1);
  BOOST_CHECK(!it2);
}

BOOST_FIXTURE_TEST_CASE(test_asignment_iterator, fixture) {
  dprocess_domain p01 = {p[0], p[1]};

  uint_sequence_dataset<>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(p01);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 4);
  BOOST_CHECK_EQUAL(it->first.at(p[0](0)), seq0(0, 0));
  BOOST_CHECK_EQUAL(it->first.at(p[1](0)), seq0(1, 0));
  BOOST_CHECK_EQUAL(it->first.at(p[0](1)), seq0(0, 1));
  BOOST_CHECK_EQUAL(it->first.at(p[1](1)), seq0(1, 1));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, p01));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(p[0](0)), seq1(0, 0));
  BOOST_CHECK_EQUAL(it->first.at(p[1](0)), seq1(1, 0));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, p01));
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

BOOST_FIXTURE_TEST_CASE(test_weight_iterator, fixture) {
  uint_sequence_dataset<>::weight_iterator it, end;
  std::tie(it, end) = ds.weights();

  BOOST_CHECK_EQUAL(*it, 0.5);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  BOOST_CHECK_EQUAL(*it, 1.0);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

BOOST_FIXTURE_TEST_CASE(test_mutation, fixture) {
  // replace the values for process 1 with all-zeros
  for (auto& s : ds({p[1]})) {
    s.first.fill(1);
    s.second = 0.3;
  }

  // check if we have stored the values
  seq0.row(1).fill(1);
  seq1.row(1).fill(1);
  BOOST_CHECK_EQUAL(ds[0].first, seq0);
  BOOST_CHECK_EQUAL(ds[0].second, 0.3);
  BOOST_CHECK_EQUAL(ds[1].first, seq1);
  BOOST_CHECK_EQUAL(ds[1].second, 0.3);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  // repeatedly shuffle and check if both permutations have the same frequency
  std::mt19937 rng;
  int norig = 0;
  int nswap = 0;
  int nbad = 0;
  std::size_t nshuffles = 500;
  for (std::size_t i = 0; i < nshuffles; ++i) {
    ds.shuffle(rng);
    if (ds.size() == 2 && ds.arguments() == p) {
      if (ds[0].first.cols() == 2 && ds[0].first == seq0 &&
          ds[1].first.cols() == 1 && ds[1].first == seq1) {
        ++norig;
      } else if (ds[0].first.cols() == 1 && ds[0].first == seq1 &&
                 ds[1].first.cols() == 2 && ds[1].first == seq0) {
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

  // load the data
  universe u;
  text_dataset_format format;
  uint_sequence_dataset<> ds;
  format.load_config(dir + "/finite_seq.cfg", u);
  load({dir + "/finite_seq0.txt", dir + "/finite_seq1.txt"}, format, ds);

  // check the sequences
  data_type seq0(2, 3), seq1(2, 4);
  seq0 << 1, 0, 2, 1, 1, 0;
  seq1 << 0, 2, 3, -1, 0, 0, -1, 1;
  BOOST_CHECK_EQUAL(ds.size(), 2);
  BOOST_CHECK_EQUAL(ds[0].first, seq0);
  BOOST_CHECK_EQUAL(ds[0].second, 1.0);
  BOOST_CHECK_EQUAL(ds[1].first, seq1);
  BOOST_CHECK_EQUAL(ds[1].second, 1.0);

  save({"finite_seq0.tmp", "finite_seq1.tmp"}, format, ds);
}