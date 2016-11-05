#define BOOST_TEST_MODULE submatrix
#include <boost/test/unit_test.hpp>

#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include "helpers.hpp"

using sptr = const std::size_t*;

namespace libgm {
  template class submatrix<real_matrix<double>, counting_iterator, sptr>;
  template class submatrix<real_matrix<float>, sptr, sptr>;
  template class submatrix<const real_matrix<double>, sptr, sptr>;
  template class submatrix<const real_matrix<float>, sptr, counting_iterator>;
}

using namespace libgm;

typedef real_matrix<double> mat_type;

BOOST_AUTO_TEST_CASE(test_operations) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  ivec rows = {1};
  ivec cols = {2, 0};
  submatrix<const mat_type, sptr, sptr> sm(m, rows, cols);

  BOOST_CHECK_EQUAL(sm.rows(), 1);
  BOOST_CHECK_EQUAL(sm.cols(), 2);
  BOOST_CHECK_EQUAL(sm.colptr(0), m.data() + 2 * 2);
  BOOST_CHECK_EQUAL(sm.colptr(1), m.data() + 2 * 0);
  BOOST_CHECK_EQUAL(sm, mat12(-1, 1));

  ivec rows1 = {1};
  ivec cols2 = {1, 2};
  BOOST_CHECK_EQUAL(submat(m, rows1, cols2), mat12(0, -1));
}

BOOST_AUTO_TEST_CASE(test_update) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  ivec row0 = {0};
  ivec row1 = {1};
  ivec cols = {0, 2};

  // first, test updates of a dense matrix
  mat_type w = submat(m, row0, cols);
  BOOST_CHECK(w.isApprox(mat12(4, 2), 1e-8));
  w += submat(m, row0, cols);
  BOOST_CHECK(w.isApprox(mat12(8, 4), 1e-8));
  w -= submat(m, row1, cols);
  BOOST_CHECK(w.isApprox(mat12(7, 5), 1e-8));

  // then, test updates of a sparse matrix
  mat_type u = mat_type::Zero(2, 3);
  submat(u, row1, cols) = w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 7, 0, 5), 1e-8));
  submat(u, row1, cols) += w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 14, 0, 10), 1e-8));
  submat(u, row0, cols) -= w;
  BOOST_CHECK(u.isApprox(mat23(-7, 0, -5, 14, 0, 10), 1e-8));
}
