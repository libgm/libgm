#define BOOST_TEST_MODULE hybrid_sequence_dataset
#include <boost/test/unit_test.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/learning/dataset/hybrid_sequence_dataset.hpp>
#include <libgm/learning/dataset/hybrid_sequence_dataset_io.hpp>

#include <random>

namespace libgm {
  template class hybrid_sequence_dataset<var, double>;
  template class hybrid_sequence_dataset<vec, double>;
}

using namespace libgm;

typedef std::pair<hybrid_matrix<>, double> sample_type;
// typedef std::pair<hybrid_assignment<var>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
// BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

struct fixture {
  universe u;
  sequence<var> a, b, c;
  sequence<var> x, y;
  domain<sequence<var> > bcy, dom;
  hybrid_matrix<> seq0, seq1;
  hybrid_sequence_dataset<var> ds;

  fixture()
    : seq0(3, 2, 2, 2), seq1(3, 1, 2, 1) {
    a = var::discrete(u, "a", 2).desc();
    b = var::discrete(u, "b", 3).desc();
    c = var::discrete(u, "c", 4).desc();
    x = var::continuous(u, "x").desc();
    y = var::continuous(u, "y").desc();
    bcy = {b, c, y};
    dom = {a, b, c, x, y};
    seq0.uint() << 0, 1, 1, 2, 2, 3;
    seq0.real() << 3, 4, 5, 6;
    seq1.uint() << 1, 2, 3;
    seq1.real() << 4, 5;
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
  hybrid_sequence_dataset<var>::const_iterator it = ds.begin();
  hybrid_sequence_dataset<var>::const_iterator end = ds.end();
  BOOST_CHECK_EQUAL(it->first, seq0);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.sample(1));
  ++it;
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.uint().rows(), 3);
    BOOST_CHECK_EQUAL(it->first.uint().cols(), 0);
    BOOST_CHECK_EQUAL(it->first.real().rows(), 2);
    BOOST_CHECK_EQUAL(it->first.real().cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);

  // indirect iteration
  // direct iteration
  const auto& cds = ds;
  std::tie(it, end) = cds.samples(bcy);
  BOOST_CHECK_EQUAL(it->first.uint(), seq0.uint().block(1, 0, 2, 2));
  BOOST_CHECK_EQUAL(it->first.real(), seq0.real().block(1, 0, 1, 2));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.sample(0, bcy));
  ++it;
  BOOST_CHECK_EQUAL(it->first.uint(), seq1.uint().block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->first.real(), seq1.real().block(1, 0, 1, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.sample(1, bcy));
  ++it;
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.uint().rows(), 2);
    BOOST_CHECK_EQUAL(it->first.uint().cols(), 0);
    BOOST_CHECK_EQUAL(it->first.real().rows(), 1);
    BOOST_CHECK_EQUAL(it->first.real().cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(test_value_iterators, fixture) {
  hybrid_sequence_dataset<var>::iterator it1, end1;
  std::tie(it1, end1) = ds.samples(dom);

  hybrid_sequence_dataset<var>::const_iterator it2 = ds.begin();
  hybrid_sequence_dataset<var>::const_iterator end2 = ds.end();

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

BOOST_FIXTURE_TEST_CASE(test_weight_iterator, fixture) {
  hybrid_sequence_dataset<var>::weight_iterator it, end;
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
  for (auto& s : ds.samples({b, x})) {
    s.first.uint().fill(1);
    s.first.real().fill(9);
    s.second = 0.3;
  }

  // check if we have stored the values
  seq0.uint().row(1).fill(1);
  seq0.real().row(0).fill(9);
  seq1.uint().row(1).fill(1);
  seq1.real().row(0).fill(9);
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
  hybrid_sequence_dataset<var> ds;
  format.load_config(dir + "/hybrid_seq.cfg", u);
  load({dir + "/hybrid_seq0.txt", dir + "/hybrid_seq1.txt"}, format, ds);

  // check the sequences
  hybrid_matrix<> seq0(2, 3, 3, 3), seq1(2, 4, 3, 4);
  seq0.uint() << 1, 0, 2, 1, 1, 0;
  seq0.real() << 0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 1.0, 0.9, 0.8;
  seq1.uint() << 0, 2, 3, -1, 0, 0, -1, 1;
  seq1.real() << 1, 2, 3, 999, 999, 999, 5, 6, 0.1, 0.2, 0.3, 0.4;
  BOOST_CHECK_EQUAL(ds.size(), 2);
  BOOST_CHECK_EQUAL(ds.sample(0).first, seq0);
  BOOST_CHECK_EQUAL(ds.sample(0).second, 1.0);

  hybrid_matrix<> sample1 = ds.sample(1).first;
  std::replace_if(sample1.real().data(),
                  sample1.real().data() + seq1.real().size(),
                  ismissing<double>, 999);
  BOOST_CHECK_EQUAL(sample1, seq1);
  BOOST_CHECK_EQUAL(ds.sample(1).second, 1.0);

  save({"hybrid_seq0.tmp", "hybrid_seq1.tmp"}, format, ds);
}
