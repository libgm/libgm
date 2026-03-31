#define BOOST_TEST_MODULE named_argument
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>

#include <sstream>
#include <stdexcept>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_make_and_identity) {
  using Arg = NamedArg<16>;

  Arg x1("x");
  Arg x2("x");
  Arg y("y");

  BOOST_CHECK(x1 == x2);
  BOOST_CHECK(x1 != y);
}

BOOST_AUTO_TEST_CASE(test_empty_name_is_valid_non_null_value) {
  using Arg = NamedArg<16>;

  Arg a("");
  BOOST_CHECK_EQUAL(std::string(a.c_str()), "");
}

BOOST_AUTO_TEST_CASE(test_length_is_checked) {
  using Arg = NamedArg<16>;

  BOOST_CHECK_THROW(Arg("this_name_is_way_too_long"), std::length_error);
}

BOOST_AUTO_TEST_CASE(test_stream_output_uses_c_string) {
  using Arg = NamedArg<16>;

  std::ostringstream out;
  out << Arg("hello");
  BOOST_CHECK_EQUAL(out.str(), "hello");
}
