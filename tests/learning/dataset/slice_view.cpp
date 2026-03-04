#define BOOST_TEST_MODULE slice_view
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/slice_view.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/learning/dataset/uint_dataset.hpp>
#include <libgm/learning/dataset/real_dataset.hpp>
#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>


typedef libgm::slice_view<libgm::uint_dataset<libgm::var>> view_type;

namespace libgm {
  template class slice_view<uint_dataset<var> >;
  template class slice_view<real_dataset<var> >;
  template class slice_view<hybrid_dataset<var> >;
  template class slice_view<uint_sequence_dataset<var> >;
  template class view_type::
    template slice_iterator<uint_dataset<var>::iterator>;
  template class view_type::
    template slice_iterator<uint_dataset<var>::const_iterator>;
}

using namespace libgm;

typedef std::pair<uint_vector, double> sample_type;
typedef probability_table<var> ptable;
BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);


BOOST_AUTO_TEST_CASE(test_accessors) {
  universe u;
  var x = var::discrete(u, "x", 3);
  var y = var::discrete(u, "y", 3);

  uint_dataset<var> ds({x, y});
  ds.insert(uint_vector({1, 2}), 0.5);
  ds.insert(uint_vector({0, 1}), 0.3);
  ds.insert(uint_vector({2, 0}), 1.0);
  ds.insert(uint_vector({1, 0}), 2.0);

  auto view = subset(ds, slice(1, 3));

  // arguments
  BOOST_CHECK_EQUAL(view.arguments(), domain<var>({x, y}));
  BOOST_CHECK_EQUAL(view.arity(), 2);
  BOOST_CHECK_EQUAL(view.size(), 3);
  BOOST_CHECK_EQUAL(view.num_slices(), 1);

  // total weight
  BOOST_CHECK_CLOSE(view.weight(), 3.3, 1e-6);

  // sample
  BOOST_CHECK_EQUAL(view.sample(0).first, uint_vector({0, 1}));
  BOOST_CHECK_EQUAL(view.sample(2).first, uint_vector({1, 0}));
  BOOST_CHECK_EQUAL(view.sample(0).second, 0.3);
  BOOST_CHECK_EQUAL(view.sample(2).second, 2.0);

  // sample over a domain
  BOOST_CHECK_EQUAL(view.sample(0, {y}).first, uint_vector({1}));
  BOOST_CHECK_EQUAL(view.sample(2, {y}).first, uint_vector({0}));
  BOOST_CHECK_EQUAL(view.sample(0, {y}).second, 0.3);
  BOOST_CHECK_EQUAL(view.sample(2, {y}).second, 2.0);

  /*
  // assignment
  std::pair<uint_assignment<var>, double> a;
  a = view.assignment(1);
  BOOST_CHECK_EQUAL(a.first.size(), 2);
  BOOST_CHECK_EQUAL(a.first.at(x), 2);
  BOOST_CHECK_EQUAL(a.first.at(y), 0);
  BOOST_CHECK_EQUAL(a.second, 1.0);

  a = view.assignment(2, {x});
  BOOST_CHECK_EQUAL(a.first.size(), 1);
  BOOST_CHECK_EQUAL(a.first.at(x), 1);
  BOOST_CHECK_EQUAL(a.second, 2.0);
  */
}


BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  var x = var::discrete(u, "x", 3);
  var y = var::discrete(u, "y", 3);

  uint_dataset<var> ds({x, y});
  ds.insert(uint_vector({1, 2}), 0.5);
  ds.insert(uint_vector({0, 1}), 0.3);
  ds.insert(uint_vector({2, 0}), 1.0);
  ds.insert(uint_vector({1, 0}), 2.0);

  view_type view = subset(ds, {slice(2, 2), slice(0, 1)});
  view_type::const_iterator it = view.begin(), end = view.end();

  // conversion
  view_type::iterator it2 = view.begin(), end2 = view.end();
  BOOST_CHECK(it == it2);
  BOOST_CHECK(end == end2);
  ++it2;
  BOOST_CHECK(it != it2);

  // absolute index 2
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first, uint_vector({2, 0}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  ++it;

  // absolute index 3
  BOOST_CHECK(it != end);
  BOOST_CHECK(it == it2);
  BOOST_CHECK_EQUAL(it->first, uint_vector({1, 0}));
  BOOST_CHECK_EQUAL(it->second, 2.0);
  BOOST_CHECK(it);
  ++it;

  // absolute index 0
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first, uint_vector({1, 2}));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK(it);
  ++it;

  // reached the end?
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

/*
BOOST_AUTO_TEST_CASE(test_assignment_iterator) {
  universe u;
  var x = var::discrete(u, "x", 3);
  var y = var::discrete(u, "y", 3);

  uint_dataset<var> ds({x, y});
  ds.insert(uint_vector({1, 2}), 0.5);
  ds.insert(uint_vector({0, 1}), 0.3);
  ds.insert(uint_vector({2, 0}), 1.0);
  ds.insert(uint_vector({1, 0}), 2.0);

  view_type view = subset(ds, {slice(2, 2), slice(1, 1)});
  view_type::assignment_iterator it, end;
  std::tie(it, end) = view.assignments({x});

  // absolute index 2
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 2);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  ++it;

  // absolute index 3
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 1);
  BOOST_CHECK_EQUAL(it->second, 2.0);
  BOOST_CHECK(it);
  ++it;

  // absolute index 1
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 0);
  BOOST_CHECK_EQUAL(it->second, 0.3);
  BOOST_CHECK(it);
  ++it;

  // reached the end?
  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}
*/

BOOST_AUTO_TEST_CASE(test_weight_iterator) {
  uint_dataset<var> ds(domain<var>{});
  ds.insert(uint_vector(), 0.5);
  ds.insert(uint_vector(), 0.3);
  ds.insert(uint_vector(), 1.0);
  ds.insert(uint_vector(), 2.0);

  view_type view = subset(ds, {slice(2, 2), slice(1, 1)});
  view_type::weight_iterator it, end;
  std::tie(it, end) = view.weights();

  // absolute index 2
  BOOST_CHECK_EQUAL(*it, 1.0);
  BOOST_CHECK(it != end);
  ++it;

  // absolute index 3
  BOOST_CHECK_EQUAL(*it, 2.0);
  BOOST_CHECK(it != end);
  ++it;

  // absolute index 1
  BOOST_CHECK_EQUAL(*it, 0.3);
  BOOST_CHECK(it != end);
  ++it;

  // reached the end?
  BOOST_CHECK(it == end);
}


BOOST_AUTO_TEST_CASE(test_reconstruction) {
  // generate some random samples
  universe u;
  std::mt19937 rng;
  domain<var> v = {
    var::discrete(u, "a", 2), var::discrete(u, "b", 2), var::discrete(u, "c", 2)
  };
  ptable f = uniform_table_generator<ptable>()(v, rng).normalize();
  auto d = f.distribution();
  uint_dataset<var> ds(v, 1000);
  for (std::size_t i = 0; i < 1000; ++i) {
    ds.insert(d(rng), 1.0);
  }
  factor_mle<ptable> mle;

  // contiguous range
  view_type view1 = subset(ds, slice(0, 500));
  BOOST_CHECK_EQUAL(ds.sample(48), view1.sample(48));
  BOOST_CHECK_EQUAL(ds.sample(99), view1.sample(99));
  BOOST_CHECK_SMALL(kl_divergence(f, mle(view1, v)), 1e-2);

  // multiple slices
  std::vector<slice> slices;
  slices.push_back(slice(100, 100));
  slices.push_back(slice(300, 400));
  view_type view2 = subset(ds, slices);
  BOOST_CHECK_EQUAL(ds.sample(140), view2.sample(40));
  BOOST_CHECK_EQUAL(ds.sample(350), view2.sample(150));
  BOOST_CHECK_SMALL(kl_divergence(f, mle(view2, v)), 1e-2);

  // verify the number of samples
  std::size_t n_mutable = 0;
  for (const auto& sample : view2) {
    ++n_mutable;
  }
  BOOST_CHECK_EQUAL(n_mutable, 500);

  std::size_t n_const = 0;
  const auto& view2c = view2;
  for (const auto& sample : view2c) {
    ++n_const;
  }
  BOOST_CHECK_EQUAL(n_const, 500);
}

