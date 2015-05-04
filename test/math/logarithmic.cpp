#define BOOST_TEST_MODULE logarithmic
#include <boost/test/unit_test.hpp>

#include <libgm/math/logarithmic.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace libgm {
  template class logarithmic<double>;
  template class logarithmic<float>;
}

BOOST_AUTO_TEST_CASE(test_operations) {
  using libgm::log_tag;
  using libgm::logd;

  // Create some values
  double x = 1.0;
  double y = 2.0;
  double z = 3.0;

  // Create some log values
  logd lx(x);
  logd ly(y);
  logd lz(z);
  logd l1(0, log_tag());

  // Check the constructors and conversion operators
  BOOST_CHECK_CLOSE(double(lx), x, 1e-10);
  BOOST_CHECK_CLOSE(double(ly), y, 1e-10);
  BOOST_CHECK_CLOSE(double(lz), z, 1e-10);
  BOOST_CHECK_CLOSE(double(l1), 1, 1e-10);

  // Check the binary operations
  BOOST_CHECK_CLOSE(double(lx + ly), 3.0, 1e-10);
  BOOST_CHECK_CLOSE(double(lz - ly), 1.0, 1e-10);
  BOOST_CHECK_CLOSE(double(ly * lz), 6.0, 1e-10);
  BOOST_CHECK_CLOSE(double(lz / ly), 1.5, 1e-10);

  // Check the in-place operations
  logd tmp;
  tmp = lx; tmp += ly; BOOST_CHECK_CLOSE(double(tmp), 3.0, 1e-10);
  tmp = lz; tmp -= ly; BOOST_CHECK_CLOSE(double(tmp), 1.0, 1e-10);
  tmp = ly; tmp *= lz; BOOST_CHECK_CLOSE(double(tmp), 6.0, 1e-10);
  tmp = lz; tmp /= ly; BOOST_CHECK_CLOSE(double(tmp), 1.5, 1e-10);

  // Check comparisons
  BOOST_CHECK(ly == logd(y));
  BOOST_CHECK(ly != logd(z));
  BOOST_CHECK(ly < lz);
  BOOST_CHECK(lz > ly);
  BOOST_CHECK(ly <= ly);
  BOOST_CHECK(ly >= ly);

  // Check I/O
  std::ostringstream out;
  out << std::setprecision(16) << ly;
  std::istringstream in(out.str());
  logd ly2;
  in >> ly2;
  BOOST_CHECK_CLOSE(double(ly), double(ly2), 1e-10);
}
