#define BOOST_TEST_MODULE exp
#include <boost/test/unit_test.hpp>

#include <libgm/math/exp.hpp>

#include <cmath>
#include <iomanip>
#include <sstream>

namespace libgm {
  template struct Exp<double>;
  template struct Exp<float>;
}

BOOST_AUTO_TEST_CASE(test_operations) {
  using libgm::Exp;

  const double x = 1.0;
  const double y = 2.0;
  const double z = 3.0;

  Exp<double> ex(std::log(x));
  Exp<double> ey(std::log(y));
  Exp<double> ez(std::log(z));
  Exp<double> e1(0.0);

  // Constructors and conversion operator.
  BOOST_CHECK_CLOSE(double(ex), x, 1e-10);
  BOOST_CHECK_CLOSE(double(ey), y, 1e-10);
  BOOST_CHECK_CLOSE(double(ez), z, 1e-10);
  BOOST_CHECK_CLOSE(double(e1), 1.0, 1e-10);

  // Binary operations.
  BOOST_CHECK_CLOSE(double(ex * ey), 2.0, 1e-10);
  BOOST_CHECK_CLOSE(double(ez / ey), 1.5, 1e-10);

  // In-place operations.
  Exp<double> tmp;
  tmp = ey; tmp *= ez; BOOST_CHECK_CLOSE(double(tmp), 6.0, 1e-10);
  tmp = ez; tmp /= ey; BOOST_CHECK_CLOSE(double(tmp), 1.5, 1e-10);

  // Comparisons.
  BOOST_CHECK(ey == Exp<double>(std::log(y)));
  BOOST_CHECK(ey != Exp<double>(std::log(z)));
  BOOST_CHECK(ey < ez);
  BOOST_CHECK(ez > ey);
  BOOST_CHECK(ey <= ey);
  BOOST_CHECK(ey >= ey);

  // Friends: pow and log.
  BOOST_CHECK_CLOSE(double(pow(ey, 3.0)), 8.0, 1e-10);
  BOOST_CHECK_CLOSE(log(ez), std::log(z), 1e-10);

  // I/O.
  std::ostringstream out;
  out << std::setprecision(16) << ey;
  std::istringstream in(out.str());
  Exp<double> ey2;
  in >> ey2;
  BOOST_CHECK_CLOSE(double(ey), double(ey2), 1e-10);
}
