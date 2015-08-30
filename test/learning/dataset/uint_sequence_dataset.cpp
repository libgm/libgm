#define BOOST_TEST_MODULE uint_sequence_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset_io.hpp>

#include <random>

namespace libgm {
  template class basic_sequence_dataset<var, uint_vector, uint_matrix, double>;
  template class basic_sequence_dataset<vec, uint_vector, uint_matrix, double>;
}

using namespace libgm;

typedef std::pair<uint_matrix, double> sample_type;
// typedef std::pair<uint_assignment<var>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
// BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

struct fixture {
  universe u;
  sequence<var> a, b, c;
  domain<sequence<var> > bc, dom;
  uint_matrix seq0, seq1;
  uint_sequence_dataset<var> ds;

  fixture()
    : seq0(3, 2), seq1(3, 1) {
    a = var::discrete(u, "a", 2).desc();
    b = var::discrete(u, "b", 3).desc();
    c = var::discrete(u, "c", 4).desc();
    bc = {b, c};
    dom = {a, b, c};
    seq0 << 0, 1, 1, 2, 2, 3;
    seq1 << 1, 2, 3;
    ds.initialize(dom);
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
  uint_sequence_dataset<var>::const_iterator it = ds.begin();
  uint_sequence_dataset<var>::const_iterator end = ds.end();
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
  std::tie(it, end) = cds.samples(bc);
  BOOST_CHECK_EQUAL(it->first, seq0.block(1, 0, 2, 2));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0, bc));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.sample(1, bc));
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
  uint_sequence_dataset<var>::iterator it1, end1;
  std::tie(it1, end1) = ds.samples({a, b, c});

  uint_sequence_dataset<var>::const_iterator it2 = ds.begin();
  uint_sequence_dataset<var>::const_iterator end2 = ds.end();

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
BOOST_FIXTURE_TEST_CASE(test_asignment_iterator, fixture) {
  uint_sequence_dataset<var>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(ab);

  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 4);
  BOOST_CHECK_EQUAL(it->first.at(a(0)), seq0(0, 0));
  BOOST_CHECK_EQUAL(it->first.at(b(0)), seq0(1, 0));
  BOOST_CHECK_EQUAL(it->first.at(a(1)), seq0(0, 1));
  BOOST_CHECK_EQUAL(it->first.at(b(1)), seq0(1, 1));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, ab));
  BOOST_CHECK(it);
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a(0)), seq1(0, 0));
  BOOST_CHECK_EQUAL(it->first.at(b(0)), seq1(1, 0));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, p01));
  BOOST_CHECK(it);
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}
*/

BOOST_FIXTURE_TEST_CASE(test_weight_iterator, fixture) {
  uint_sequence_dataset<var>::weight_iterator it, end;
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
  for (auto& s : ds.samples({b})) {
    s.first.fill(1);
    s.second = 0.3;
  }

  // check if we have stored the values
  seq0.row(1).fill(1);
  seq1.row(1).fill(1);
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
    if (ds.size() == 2 && ds.arguments() == dom) {
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
  text_dataset_format<var> format;
  uint_sequence_dataset<var> ds;
  format.load_config(dir + "/uint_seq.cfg", u);
  load({dir + "/uint_seq0.txt", dir + "/uint_seq1.txt"}, format, ds);

  // check the sequences
  uint_matrix seq0(2, 3), seq1(2, 4);
  seq0 << 1, 0, 2, 1, 1, 0;
  seq1 << 0, 2, 3, -1, 0, 0, -1, 1;
  BOOST_CHECK_EQUAL(ds.size(), 2);
  BOOST_CHECK_EQUAL(ds.sample(0).first, seq0);
  BOOST_CHECK_EQUAL(ds.sample(0).second, 1.0);
  BOOST_CHECK_EQUAL(ds.sample(1).first, seq1);
  BOOST_CHECK_EQUAL(ds.sample(1).second, 1.0);

  save({"uint_seq0.tmp", "uint_seq1.tmp"}, format, ds);
}
