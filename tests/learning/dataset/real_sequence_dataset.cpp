#define BOOST_TEST_MODULE real_sequence_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/real_sequence_dataset.hpp>
#include <libgm/learning/dataset/real_sequence_dataset_io.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>

#include <random>

namespace libgm {
  template class basic_sequence_dataset<var, dense_vector<>, dense_matrix<>, double>;
  template class basic_sequence_dataset<vec, dense_vector<>, dense_matrix<>, double>;
}

using namespace libgm;

typedef std::pair<dense_matrix<>, double> sample_type;
// typedef std::pair<real_assignment<double>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
// BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

struct fixture {
  universe u;
  sequence<vec> a, b;
  domain<sequence<vec> > ab;
  dense_matrix<> seq0, seq1;
  real_sequence_dataset<vec> ds;

  fixture()
    : seq0(3, 2), seq1(3, 1) {
    a = vec::continuous(u, "a", 1).desc();
    b = vec::continuous(u, "b", 2).desc();
    ab = {a, b};
    seq0 << 0, 1, 1, 2, 2, 3;
    seq1 << 1, 2, 3;
    ds.initialize(ab);
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
  real_sequence_dataset<vec>::const_iterator it = ds.begin();
  real_sequence_dataset<vec>::const_iterator end = ds.end();
  BOOST_CHECK_EQUAL(it->first, seq0);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.sample(1));
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
  const auto& cds = ds;
  std::tie(it, end) = cds.samples({b});
  BOOST_CHECK_EQUAL(it->first, seq0.block(1, 0, 2, 2));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0, {b}));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.sample(1, {b}));
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
  real_sequence_dataset<vec>::iterator it1, end1;
  std::tie(it1, end1) = ds.samples(ab);

  real_sequence_dataset<vec>::const_iterator it2 = ds.begin();
  real_sequence_dataset<vec>::const_iterator end2 = ds.end();

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

/*
BOOST_FIXTURE_TEST_CASE(test_assignment_iterator, fixture) {
  real_sequence_dataset<vec>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(ab);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 4);
  BOOST_CHECK_EQUAL(it->first.at(a(0)), seq0.block(0, 0, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(b(0)), seq0.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->first.at(a(1)), seq0.block(0, 1, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(b(1)), seq0.block(1, 1, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, ab));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a(0)), seq1.block(0, 0, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(b(0)), seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, ab));
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}
*/

BOOST_FIXTURE_TEST_CASE(test_weight_iterator, fixture) {
  real_sequence_dataset<vec>::weight_iterator it, end;
  std::tie(it, end) = ds.weights();

  // check the first sample
  BOOST_CHECK_EQUAL(*it, 0.5);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(*it, 1.0);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

BOOST_FIXTURE_TEST_CASE(test_mutation, fixture) {
  // replace the values for process 1 with all-zeros
  for (auto& s : ds.samples({b})) {
    s.first.row(0).fill(0.1);
    s.second = 0.3;
  }

  // check if we have stored the values
  seq0.row(1).fill(0.1);
  seq1.row(1).fill(0.1);
  BOOST_CHECK_EQUAL(ds.sample(0).first, seq0);
  BOOST_CHECK_EQUAL(ds.sample(0).second, 0.3);
  BOOST_CHECK_EQUAL(ds.sample(1).first, seq1);
  BOOST_CHECK_EQUAL(ds.sample(1).second, 0.3);
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
    if (ds.size() == 2 && ds.arguments() == ab) {
      if (ds.sample(0).first.cols() == 2 && ds.sample(0).first == seq0 &&
          ds.sample(1).first.cols() == 1 && ds.sample(1).first == seq1) {
        ++norig;
      } else if (ds.sample(0).first.cols() == 1 && ds.sample(0).first == seq1 &&
                 ds.sample(1).first.cols() == 2 && ds.sample(1).first == seq0) {
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
  text_dataset_format<vec> format;
  real_sequence_dataset<vec> ds;
  format.load_config(dir + "/real_seq.cfg", u);
  load({dir + "/real_seq0.txt", dir + "/real_seq1.txt"}, format, ds);

  // check the sequences
  dense_matrix<> seq0(3, 3), seq1(3, 2);
  seq0 << 0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 1.0, 0.9, 0.8;
  seq1 << -0.1, -0.2, -0.2, -0.3, 0.8, 0.9;
  BOOST_CHECK_EQUAL(ds.size(), 2);
  BOOST_CHECK_EQUAL(ds.sample(0).first, seq0);
  BOOST_CHECK_EQUAL(ds.sample(0).second, 1.0);
  BOOST_CHECK_EQUAL(ds.sample(1).first, seq1);
  BOOST_CHECK_EQUAL(ds.sample(1).second, 1.0);

  save({"real_seq0.tmp", "real_seq1.tmp"}, format, ds);
}
