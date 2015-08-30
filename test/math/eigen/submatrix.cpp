#define BOOST_TEST_MODULE submatrix
#include <boost/test/unit_test.hpp>

#include <libgm/math/eigen/real.hpp>
#include <libgm/math/eigen/submatrix.hpp>

#include "helpers.hpp"

namespace libgm {
  template class submatrix<real_matrix<double> >;
  template class submatrix<real_matrix<float> >;
  template class submatrix<const real_matrix<double> >;
  template class submatrix<const real_matrix<float> >;
}

using namespace libgm;

typedef real_matrix<double> mat_type;

BOOST_AUTO_TEST_CASE(test_operations) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  std::vector<std::size_t> rows = {1};
  std::vector<std::size_t> cols = {2, 0};
  submatrix<const mat_type> sm(m, rows, cols);

  BOOST_CHECK_EQUAL(sm.rows(), 1);
  BOOST_CHECK_EQUAL(sm.cols(), 2);
  BOOST_CHECK(sm.row_contiguous());
  BOOST_CHECK(!sm.col_contiguous());
  BOOST_CHECK(!sm.contiguous());
  BOOST_CHECK_EQUAL(sm.colptr(0), m.data() + 2 * 2);
  BOOST_CHECK_EQUAL(sm.colptr(1), m.data() + 2 * 0);

  BOOST_CHECK_EQUAL(sm.ref(), mat12(-1, 1));

  std::vector<std::size_t> rows1 = {1};
  std::vector<std::size_t> cols2 = {1, 2};
  BOOST_CHECK_EQUAL(submat(m, rows1, cols2).ref(), mat12(0, -1));
}

BOOST_AUTO_TEST_CASE(test_update_block) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  std::vector<std::size_t> row0 = {0};
  std::vector<std::size_t> row1 = {1};
  std::vector<std::size_t> cols = {1, 2};

  // first, test updates of a dense matrix
  mat_type w;
  submat(m, row0, cols).eval_to(w);
  BOOST_CHECK(w.isApprox(mat12(3, 2), 1e-8));
  w += submat(m, row0, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 4), 1e-8));
  w -= submat(m, row1, cols);
  BOOST_CHECK(w.isApprox(mat12(6, 5), 1e-8));

  // then, test updates of a sparse matrix
  mat_type u = mat_type::Zero(2, 3);
  submat(u, row1, cols) = w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 6, 5), 1e-8));
  submat(u, row1, cols) += w;
  BOOST_CHECK(u.isApprox(mat23(0, 0, 0, 0, 12, 10), 1e-8));
  submat(u, row0, cols) -= w;
  BOOST_CHECK(u.isApprox(mat23(0, -6, -5, 0, 12, 10), 1e-8));
}

BOOST_AUTO_TEST_CASE(test_update_plain) {
  mat_type m = mat23(4, 3, 2, 1, 0, -1);
  std::vector<std::size_t> row0 = {0};
  std::vector<std::size_t> row1 = {1};
  std::vector<std::size_t> cols = {0, 2};

  // first, test updates of a dense matrix
  mat_type w;
  submat(m, row0, cols).eval_to(w);
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
