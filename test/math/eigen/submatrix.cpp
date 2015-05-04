#define BOOST_TEST_MODULE submatrix
#include <boost/test/unit_test.hpp>

#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include "helpers.hpp"

namespace libgm {
  template class submatrix<dynamic_matrix<double> >;
  template class submatrix<dynamic_matrix<float> >;
  template class submatrix<const dynamic_matrix<double> >;
  template class submatrix<const dynamic_matrix<float> >;
}

using namespace libgm;

typedef dynamic_matrix<double> mat_type;

BOOST_AUTO_TEST_CASE(test_operations) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  matrix_index rows = {1};
  matrix_index cols = {2, 0};
  submatrix<const mat_type> sm(m, rows, cols);
  
  BOOST_CHECK_EQUAL(sm.rows(), 1);
  BOOST_CHECK_EQUAL(sm.cols(), 2);
  BOOST_CHECK(!sm.contiguous());
  BOOST_CHECK_EQUAL(sm.row_index(0), 1);
  BOOST_CHECK_EQUAL(sm.col_index(1), 0);

  BOOST_CHECK_EQUAL(sm.plain(), mat12(-1, 1));

  matrix_index rows1(1, 1);
  matrix_index cols2(1, 2);
  BOOST_CHECK_EQUAL(submat(m, rows1, cols2).block(),mat12(0, -1));
}

BOOST_AUTO_TEST_CASE(test_update_block) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  matrix_index row0(0, 1);
  matrix_index row1(1, 1);
  matrix_index cols(1, 2);
  
  // first, test updates of a dense matrix
  mat_type w;
  set(w, submat(m, row0, cols));
  BOOST_CHECK(w.isApprox(mat12(3, 2), 1e-8));
  w += submat(m, row0, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 4), 1e-8));
  w -= submat(m, row1, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 5), 1e-8));

  // then, test updates of a sparse matrix
  mat_type u = mat_type::Zero(2, 3);
  set(submat(u, row1, cols), w);
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 6, 5), 1e-8));
  submat(u, row1, cols) += w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 12, 10), 1e-8));
  submat(u, row0, cols) -= w;
  BOOST_CHECK(u.isApprox(mat23(0, -6, -5, 0, 12, 10), 1e-8));
}

BOOST_AUTO_TEST_CASE(test_update_plain) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  matrix_index row0 = {0};
  matrix_index row1 = {1};
  matrix_index cols = {1, 2};
  
  // first, test updates of a dense matrix
  mat_type w;
  set(w, submat(m, row0, cols));
  BOOST_CHECK(w.isApprox(mat12(3, 2), 1e-8));
  w += submat(m, row0, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 4), 1e-8));
  w -= submat(m, row1, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 5), 1e-8));

  // then, test updates of a sparse matrix
  mat_type u = mat_type::Zero(2, 3);
  set(submat(u, row1, cols), w);
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 6, 5), 1e-8));
  submat(u, row1, cols) += w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 12, 10), 1e-8));
  submat(u, row0, cols) -= w;
  BOOST_CHECK(u.isApprox(mat23(0, -6, -5, 0, 12, 10), 1e-8));
}
