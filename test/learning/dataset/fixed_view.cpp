#define BOOST_TEST_MODULE fixed_view
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/fixed_view.hpp>

#include <libgm/argument/universe.hpp>
#include <libgm/argument/var.hpp>
#include <libgm/argument/vec.hpp>
#include <libgm/learning/dataset/hybrid_sequence_dataset.hpp>
#include <libgm/learning/dataset/real_sequence_dataset.hpp>
#include <libgm/learning/dataset/uint_sequence_dataset.hpp>
#include <libgm/learning/dataset/slice_view.hpp>

#include "../../math/eigen/helpers.hpp"

namespace libgm {
  template class fixed_view<uint_sequence_dataset<var> >;
  template class fixed_view<real_sequence_dataset<vec> >;
  template class fixed_view<hybrid_sequence_dataset<var> >;
  template class fixed_view<slice_view<uint_sequence_dataset<var> > >;
}

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(uint_vector);

// Univariate discrete fixed views
//============================================================================

struct var_fixture {
  universe u;
  sequence<var> a;
  sequence<var> b;
  var a0;
  var a1;
  var b0;
  uint_sequence_dataset<var> ds;
  typedef fixed_view<uint_sequence_dataset<var> > view_type;

  var_fixture() {
    a = var::discrete(u, "a", 5).desc();
    b = var::discrete(u, "b", 5).desc();
    a0 = a(0);
    a1 = a(1);
    b0 = b(0);

    ds.initialize({a, b});

    uint_matrix values;
    values.resize(2, 3);
    values << 0, 1, 2, 3, 4, 1;
    ds.insert(values, 1.0);

    values.resize(2, 1);
    values << 3, 2;
    ds.insert(values, 2.0);

    values.resize(2, 2);
    values << 2, 3, 1, 0;
    ds.insert(values, 3.0);

    values.resize(2, 0);
    ds.insert(values, 0.5);
  }
};

BOOST_FIXTURE_TEST_CASE(test_accessors, var_fixture) {
  // view over a single step
  view_type view1(&ds, 1, 1);
  domain<var> args = { a(0), b(0) };
  BOOST_CHECK_EQUAL(view1.arguments(), args);
  BOOST_CHECK_EQUAL(view1.arity(), 2);
  BOOST_CHECK_EQUAL(view1.size(), 2);
  BOOST_CHECK(!view1.empty());
  BOOST_CHECK_CLOSE(view1.weight(), 4.0, 1e-6);
  std::cout << view1 << std::endl;

  // view over two steps
  view_type view2(&ds, 1, 2);
  args.insert(args.end(), { a(1), b(1) });
  BOOST_CHECK_EQUAL(view2.arguments(), args);
  BOOST_CHECK_EQUAL(view2.arity(), 4);
  BOOST_CHECK_EQUAL(view2.size(), 1);
  BOOST_CHECK(!view2.empty());
  BOOST_CHECK_CLOSE(view2.weight(), 1.0, 1e-6);

  // view over three steps
  view_type view3(&ds, 1, 3);
  args.insert(args.end(), { a(2), b(2) });
  BOOST_CHECK_EQUAL(view3.arguments(), args);
  BOOST_CHECK_EQUAL(view3.arity(), 6);
  BOOST_CHECK_EQUAL(view3.size(), 0);
  BOOST_CHECK(view3.empty());
  BOOST_CHECK_EQUAL(view3.weight(), 0.0);
}

BOOST_FIXTURE_TEST_CASE(test_const_iterator, var_fixture) {
  // view over a single step, iterate over all variables
  view_type view1(&ds, 1, 1);
  view_type::const_iterator it = view1.begin();
  view_type::const_iterator end = view1.end();

  BOOST_CHECK_EQUAL(it->first, uint_vector({1, 4}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(it->first, uint_vector({3, 0}));
  BOOST_CHECK_EQUAL(it->second, 3.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(!it);
  BOOST_CHECK(it == end);

  // view over two steps, iterate over a subset of variables
  view_type view2(&ds, 1, 2);
  std::tie(it, end) = view2.samples({ a1, a0, b0 });

  BOOST_CHECK_EQUAL(it->first, uint_vector({2, 1, 4}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(!it);
  BOOST_CHECK(it == end);
}

/*
BOOST_FIXTURE_TEST_CASE(test_assignment_iterator, var_fixture) {
  // view over a single step, iterate over all variables
  view_type view1(&ds, 1, 1);
  view_type::assignment_iterator it, end;
  std::tie(it, end) = view1.assignments();

  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 1);
  BOOST_CHECK_EQUAL(it->first.at(b0), 4);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 3);
  BOOST_CHECK_EQUAL(it->first.at(b0), 0);
  BOOST_CHECK_EQUAL(it->second, 3.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(!it);
  BOOST_CHECK(it == end);

  // view over two steps, iterate over a subset of variables
  view_type view2(&ds, 1, 2);
  std::tie(it, end) = view2.assignments({ a1, a0, b0 });

  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(a1), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 1);
  BOOST_CHECK_EQUAL(it->first.at(b0), 4);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(it);
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(!it);
  BOOST_CHECK(it == end);
}
*/

BOOST_FIXTURE_TEST_CASE(test_weight_iterator, var_fixture) {
  // view over a single step
  view_type view1(&ds, 1, 1);
  view_type::weight_iterator it, end;
  std::tie(it, end) = view1.weights();

  BOOST_CHECK_EQUAL(*it, 1.0);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  BOOST_CHECK_EQUAL(*it, 3.0);
  BOOST_CHECK(it != end);
  BOOST_CHECK(it);
  ++it;

  BOOST_CHECK(it == end);
  BOOST_CHECK(!it);
}

BOOST_FIXTURE_TEST_CASE(test_finite_access, var_fixture) {
  // view over a single step, access all variables
  view_type view1(&ds, 1, 1);
  BOOST_CHECK_EQUAL(view1.sample(0).first, uint_vector({1, 4}));
  BOOST_CHECK_EQUAL(view1.sample(1).first, uint_vector({3, 0}));
  BOOST_CHECK_EQUAL(view1.sample(0).second, 1.0);
  BOOST_CHECK_EQUAL(view1.sample(1).second, 3.0);
//   BOOST_CHECK_EQUAL(view1.assignment(1).first,
//                     uint_assignment<var>({{a0, 3}, {b0, 0}}));
//   BOOST_CHECK_EQUAL(view1.assignment(1).second, 3.0);

  // view over two steps, access a subset of variables
  view_type view2(&ds, 1, 2);
  domain<var> dom = {a1, b0};
  BOOST_CHECK_EQUAL(view2.sample(0, dom).first, uint_vector({2, 4}));
  BOOST_CHECK_EQUAL(view2.sample(0, dom).second, 1.0);
//   BOOST_CHECK_EQUAL(view2.assignment(0, dom).first,
//                     uint_assignment<var>({{a1, 2}, {b0, 4}}));
//   BOOST_CHECK_EQUAL(view2.assignment(0, dom).second, 1.0);
}

// Continuous multivariate fixed views
//============================================================================

struct vec_fixture {
  universe u;
  sequence<vec> a;
  sequence<vec> b;
  vec a0;
  vec a1;
  vec b0;
  real_sequence_dataset<vec> ds;
  typedef fixed_view<real_sequence_dataset<vec> > view_type;

  vec_fixture() {
    a = vec::continuous(u, "a", 1).desc();
    b = vec::continuous(u, "b", 2).desc();
    a0 = a(0);
    a1 = a(1);
    b0 = b(0);

    ds.initialize({a, b});

    dense_matrix<> values;
    values.resize(3, 3);
    values << 0, 1, 2, 3, 4, 1, 6, 7, 8;
    ds.insert(values, 1.0);

    values.resize(3, 1);
    values << 3, 2, 1;
    ds.insert(values, 2.0);

    values.resize(3, 2);
    values << 2, 3, 1, 0, 8, 9;
    ds.insert(values, 3.0);

    values.resize(3, 0);
    ds.insert(values, 0.5);
  }
};

BOOST_FIXTURE_TEST_CASE(test_vector_access, vec_fixture) {
  // view over a single step, access all variables
  view_type view1(&ds, 1, 1);
  BOOST_CHECK_EQUAL(view1.sample(0).first, vec3(1, 4, 7));
  BOOST_CHECK_EQUAL(view1.sample(1).first, vec3(3, 0, 9));
  BOOST_CHECK_EQUAL(view1.sample(0).second, 1.0);
  BOOST_CHECK_EQUAL(view1.sample(1).second, 3.0);
//   BOOST_CHECK_EQUAL(view1.assignment(1).first,
//                     real_assignment<vec>({{a0, vec1(3)}, {b0, vec2(0, 9)}}));
//   BOOST_CHECK_EQUAL(view1.assignment(1).second, 3.0);

  // view over two steps, access a subset of variables
  view_type view2(&ds, 1, 2);
  domain<vec> dom = {a1, b0};
  BOOST_CHECK_EQUAL(view2.sample(0, dom).first, vec3(2, 4, 7));
  BOOST_CHECK_EQUAL(view2.sample(0, dom).second, 1.0);
//   BOOST_CHECK_EQUAL(view2.assignment(0, dom).first,
//                     real_assignment<vec>({{a1, vec1(2)}, {b0, vec2(4, 7)}}));
//   BOOST_CHECK_EQUAL(view2.assignment(0, dom).second, 1.0);
}
