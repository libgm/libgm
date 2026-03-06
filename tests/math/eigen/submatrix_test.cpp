#define BOOST_TEST_MODULE submatrix
#include <boost/test/unit_test.hpp>

#include <libgm/argument/span.hpp>
#include <libgm/math/eigen/dense.hpp>
#include <libgm/math/eigen/submatrix.hpp>

using namespace libgm;

using Mat = Matrix<double>;

BOOST_AUTO_TEST_CASE(test_operations) {
  Mat m{{4, 3, 2}, {1, 0, -1}};
  Spans rows = {{1, 1}};
  Spans cols = {{2, 1}, {0, 1}}; // cols [2,0]

  auto sm = sub(m, rows, cols);
  BOOST_CHECK_EQUAL(sm.rows(), 1);
  BOOST_CHECK_EQUAL(sm.cols(), 2);
  BOOST_CHECK((Mat(sm)).isApprox(Mat{{-1, 1}}, 1e-8));

  Spans cols2 = {{1, 2}}; // cols [1,2]
  BOOST_CHECK((Mat(sub(m, rows, cols2))).isApprox(Mat{{0, -1}}, 1e-8));

  // Also exercise length>1 spans for both rows and cols.
  Mat n{{0, 1, 2, 3, 4},
        {10, 11, 12, 13, 14},
        {20, 21, 22, 23, 24},
        {30, 31, 32, 33, 34}};
  Spans rows3 = {{1, 2}}; // rows [1,2]
  Spans cols3 = {{2, 3}}; // cols [2,3,4]
  BOOST_CHECK((Mat(sub(n, rows3, cols3))).isApprox(Mat{{12, 13, 14}, {22, 23, 24}}, 1e-8));
}

BOOST_AUTO_TEST_CASE(test_update) {
  Mat m{{4, 3, 2}, {1, 0, -1}};
  Spans row0 = {{0, 1}};
  Spans row1 = {{1, 1}};
  Spans cols = {{0, 1}, {2, 1}}; // cols [0,2]

  Mat w = sub(m, row0, cols);
  BOOST_CHECK(w.isApprox(Mat{{4, 2}}, 1e-8));
  w += sub(m, row0, cols);
  BOOST_CHECK(w.isApprox(Mat{{8, 4}}, 1e-8));
  w -= sub(m, row1, cols);
  BOOST_CHECK(w.isApprox(Mat{{7, 5}}, 1e-8));

  Mat u = Mat::Zero(2, 3);
  sub(u, row1, cols) = w;
  BOOST_CHECK(u.isApprox(Mat{{0, 0, 0}, {7, 0, 5}}, 1e-8));
  sub(u, row1, cols) += w;
  BOOST_CHECK(u.isApprox(Mat{{0, 0, 0}, {14, 0, 10}}, 1e-8));
  sub(u, row0, cols) -= w;
  BOOST_CHECK(u.isApprox(Mat{{-7, 0, -5}, {14, 0, 10}}, 1e-8));

  // Exercise block updates where both row and col spans have length > 1.
  Mat n{{0, 1, 2, 3, 4},
        {10, 11, 12, 13, 14},
        {20, 21, 22, 23, 24},
        {30, 31, 32, 33, 34}};
  Spans rowsb = {{1, 2}}; // rows [1,2]
  Spans colsb = {{2, 2}}; // cols [2,3]

  Mat b = sub(n, rowsb, colsb);
  BOOST_CHECK(b.isApprox(Mat{{12, 13}, {22, 23}}, 1e-8));

  b += Mat{{1, 2}, {3, 4}};
  BOOST_CHECK(b.isApprox(Mat{{13, 15}, {25, 27}}, 1e-8));

  sub(n, rowsb, colsb) = b;
  BOOST_CHECK(n.isApprox(Mat{{0, 1, 2, 3, 4},
                             {10, 11, 13, 15, 14},
                             {20, 21, 25, 27, 24},
                             {30, 31, 32, 33, 34}}, 1e-8));
}
