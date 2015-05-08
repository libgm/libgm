#define BOOST_TEST_MODULE slice_view
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/slice_view.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/factor/probability_table.hpp>
#include <libgm/factor/random/uniform_table_generator.hpp>
#include <libgm/learning/dataset/finite_dataset.hpp>
#include <libgm/learning/dataset/vector_dataset.hpp>
#include <libgm/learning/dataset/hybrid_dataset.hpp>
#include <libgm/learning/dataset/finite_sequence_dataset.hpp>
#include <libgm/learning/parameter/factor_mle.hpp>


typedef libgm::slice_view<libgm::finite_dataset<>> view_type;

namespace libgm {
  template class slice_view<finite_dataset<> >;
  template class slice_view<vector_dataset<> >;
  template class slice_view<hybrid_dataset<> >;
  template class slice_view<finite_sequence_dataset<> >;
  template class view_type::
    template slice_iterator<finite_dataset<>::iterator>;
  template class view_type::
    template slice_iterator<finite_dataset<>::const_iterator>;
  template class view_type::
    template slice_iterator<finite_dataset<>::assignment_iterator>;
}

using namespace libgm;

typedef std::pair<finite_index, double> sample_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_index);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);


BOOST_AUTO_TEST_CASE(test_accessors) {
  universe u;
  variable x = u.new_finite_variable("x", 3);
  variable y = u.new_finite_variable("y", 3);

  finite_dataset<> ds({x, y});
  ds.insert(finite_index({1, 2}), 0.5);
  ds.insert(finite_index({0, 1}), 0.3);
  ds.insert(finite_index({2, 0}), 1.0);
  ds.insert(finite_index({1, 0}), 2.0);

  auto view = subset(ds, slice(1, 3));

  // arguments
  BOOST_CHECK_EQUAL(view.arguments(), domain({x, y}));
  BOOST_CHECK_EQUAL(view.arity(), 2);
  BOOST_CHECK_EQUAL(view.size(), 3);
  BOOST_CHECK_EQUAL(view.num_slices(), 1);

  // operator[]
  BOOST_CHECK_EQUAL(view[0].first, finite_index({0, 1}));
  BOOST_CHECK_EQUAL(view[2].first, finite_index({1, 0}));
  BOOST_CHECK_EQUAL(view[0].second, 0.3);
  BOOST_CHECK_EQUAL(view[2].second, 2.0);

  // operator()
  BOOST_CHECK_EQUAL(view(0, {y}).first, finite_index({1}));
  BOOST_CHECK_EQUAL(view(2, {y}).first, finite_index({0}));
  BOOST_CHECK_EQUAL(view(0, {y}).second, 0.3);
  BOOST_CHECK_EQUAL(view(2, {y}).second, 2.0);

  // assignment
  std::pair<finite_assignment<>, double> a;
  a = view.assignment(1);
  BOOST_CHECK_EQUAL(a.first.size(), 2);
  BOOST_CHECK_EQUAL(a.first.at(x), 2);
  BOOST_CHECK_EQUAL(a.first.at(y), 0);
  BOOST_CHECK_EQUAL(a.second, 1.0);

  a = view.assignment(2, {x});
  BOOST_CHECK_EQUAL(a.first.size(), 1);
  BOOST_CHECK_EQUAL(a.first.at(x), 1);
  BOOST_CHECK_EQUAL(a.second, 2.0);
}


BOOST_AUTO_TEST_CASE(test_value_iterators) {
  universe u;
  variable x = u.new_finite_variable("x", 3);
  variable y = u.new_finite_variable("y", 3);

  finite_dataset<> ds({x, y});
  ds.insert(finite_index({1, 2}), 0.5);
  ds.insert(finite_index({0, 1}), 0.3);
  ds.insert(finite_index({2, 0}), 1.0);
  ds.insert(finite_index({1, 0}), 2.0);

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
  BOOST_CHECK_EQUAL(it->first, finite_index({2, 0}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  ++it;

  // absolute index 3
  BOOST_CHECK(it != end);
  BOOST_CHECK(it == it2);
  BOOST_CHECK_EQUAL(it->first, finite_index({1, 0}));
  BOOST_CHECK_EQUAL(it->second, 2.0);
  ++it;

  // absolute index 0
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first, finite_index({1, 2}));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  ++it;

  // reached the end?
  BOOST_CHECK(it == end);
}


BOOST_AUTO_TEST_CASE(test_assignment_iterator) {
  universe u;
  variable x = u.new_finite_variable("x", 3);
  variable y = u.new_finite_variable("y", 3);

  finite_dataset<> ds({x, y});
  ds.insert(finite_index({1, 2}), 0.5);
  ds.insert(finite_index({0, 1}), 0.3);
  ds.insert(finite_index({2, 0}), 1.0);
  ds.insert(finite_index({1, 0}), 2.0);

  view_type view = subset(ds, {slice(2, 2), slice(1, 1)});
  view_type::assignment_iterator it, end;
  std::tie(it, end) = view.assignments({x});

  // absolute index 2
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 2);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  ++it;

  // absolute index 3
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 1);
  BOOST_CHECK_EQUAL(it->second, 2.0);
  ++it;

  // absolute index 1
  BOOST_CHECK(it != end);
  BOOST_CHECK_EQUAL(it->first.size(), 1);
  BOOST_CHECK_EQUAL(it->first.at(x), 0);
  BOOST_CHECK_EQUAL(it->second, 0.3);
  ++it;

  // reached the end?
  BOOST_CHECK(it == end);
}


BOOST_AUTO_TEST_CASE(test_reconstruction) {
  // generate some random samples
  universe u;
  std::mt19937 rng;
  domain v = u.new_finite_variables(3, "v", 2);
  ptable f = uniform_table_generator<ptable>()(v, rng).normalize();
  auto d = f.distribution();
  finite_dataset<> ds(v, 1000);
  for (std::size_t i = 0; i < 1000; ++i) {
    ds.insert(d(rng), 1.0);
  }
  factor_mle<ptable> mle;

  // contiguous range
  view_type view1 = subset(ds, slice(0, 500));
  BOOST_CHECK_EQUAL(ds[48], view1[48]);
  BOOST_CHECK_EQUAL(ds[99], view1[99]);
  BOOST_CHECK_SMALL(kl_divergence(f, mle(view1, v)), 1e-2);

  // multiple slices
  std::vector<slice> slices;
  slices.push_back(slice(100, 100));
  slices.push_back(slice(300, 400));
  view_type view2 = subset(ds, slices);
  BOOST_CHECK_EQUAL(ds[140], view2[40]);
  BOOST_CHECK_EQUAL(ds[350], view2[150]);
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

