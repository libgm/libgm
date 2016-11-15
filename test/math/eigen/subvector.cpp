#define BOOST_TEST_MODULE subvector
#include <boost/test/unit_test.hpp>

#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/subvector.hpp>

#include "helpers.hpp"

namespace libgm {
  template class subvector<dense_vector<double>, const std::size_t*>;
  template class subvector<dense_vector<float>, const std::size_t*>;
  template class subvector<const dense_vector<double>, const std::size_t*>;
  template class subvector<const dense_vector<float>, const std::size_t*>;
}

using namespace libgm;

typedef dense_vector<double> vec_type;

BOOST_AUTO_TEST_CASE(test_operations) {
  vec_type v = vec4(4, 3, 2, 1);
  ivec index = {2, 0};
  subvector<const vec_type, const std::size_t*> sv(v, index);

  BOOST_CHECK_EQUAL(sv.rows(), 2);
  BOOST_CHECK_EQUAL(sv.cols(), 1);
  BOOST_CHECK_EQUAL(sv, vec2(2, 4));

  ivec index1 = {1};
  BOOST_CHECK_EQUAL(subvec(v, index1), vec1(3));

  vec_type w = vec2(-1, 1);
  BOOST_CHECK_CLOSE(sv.dot(w), 2, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_update) {
  const vec_type v = vec4(4, 3, 2, 1);
  ivec i1 = {0, 2};
  ivec i2 = {2, 3};

  // first, test updates of a dense vector
  vec_type w = subvec(v, i1);
  BOOST_CHECK(w.isApprox(vec2(4, 2), 1e-8));
  w += subvec(v, i1);
  BOOST_CHECK(w.isApprox(vec2(8, 4), 1e-8));
  w -= subvec(v, i2);
  BOOST_CHECK(w.isApprox(vec2(6, 3), 1e-8));

  // then, test updates of a sparse vector
  vec_type u = vec4(0, 0, 0, 0);
  subvec(u, i1) = w;
  BOOST_CHECK(u.isApprox(vec4(6, 0, 3, 0), 1e-8));
  subvec(u, i1) += w;
  BOOST_CHECK(u.isApprox(vec4(12, 0, 6, 0), 1e-8));
  subvec(u, i2) -= w;
  BOOST_CHECK(u.isApprox(vec4(12, 0, 0, -3), 1e-8));
}
