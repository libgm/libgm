#define BOOST_TEST_MODULE string_functions
#include <boost/test/unit_test.hpp>

#include <libgm/parser/string_functions.hpp>

#include <boost/array.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace libgm;

typedef std::pair<std::string, std::string> string_pair;
BOOST_TEST_DONT_PRINT_LOG_VALUE(string_pair);

BOOST_AUTO_TEST_CASE(test_trim) {
  BOOST_CHECK_EQUAL(trim("  abraca dabra\t"), "abraca dabra");
  BOOST_CHECK_EQUAL(trim("  \t \n \v \f \r"), "");
}

BOOST_AUTO_TEST_CASE(test_upper_lower) {
  BOOST_CHECK_EQUAL(tolower(" ABCd1"), " abcd1");
  BOOST_CHECK_EQUAL(toupper(" ABCd1"), " ABCD1");

  std::string str = "abczQ";
  toupper_inplace(str);
  BOOST_CHECK_EQUAL(str, "ABCZQ");

  tolower_inplace(str);
  BOOST_CHECK_EQUAL(str, "abczq");
}

BOOST_AUTO_TEST_CASE(test_swap_chars) {
  std::string str = "_hello_world_";
  swap_characters_inplace(str, '_', ' ');
  BOOST_CHECK_EQUAL(str, " hello world ");

  str = "/check/this/out!";
  nonalnum_to_char_inplace(str, ' ');
  BOOST_CHECK_EQUAL(str, " check this out ");
}

BOOST_AUTO_TEST_CASE(test_parse_escaped) {
  BOOST_CHECK_EQUAL(parse_escaped("plain text"), "plain text");
  BOOST_CHECK_EQUAL(parse_escaped("\"quoted text\""), "quoted text");
  BOOST_CHECK_EQUAL(parse_escaped("\"text\\twith\\ttabs\""), "text\twith\ttabs");
  BOOST_CHECK_THROW(parse_escaped("\"missing quote"), std::runtime_error);
  BOOST_CHECK_THROW(parse_escaped("\"missing escape\\"), std::runtime_error);
  BOOST_CHECK_THROW(parse_escaped("\"extra characters\" "), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_escape_string) {
  BOOST_CHECK_EQUAL(escape_string("plain text"), "\"plain text\"");
  BOOST_CHECK_EQUAL(escape_string("text\twith\ttabs"), "\"text\\twith\\ttabs\"");
  BOOST_CHECK_EQUAL(escape_string("text\"with\"quotes"), "\"text\\\"with\\\"quotes\"");
}

BOOST_AUTO_TEST_CASE(test_conversions) {
  BOOST_CHECK_EQUAL(to_string(2), "2");
  BOOST_CHECK_EQUAL(to_string(1.5), "1.5");
  BOOST_CHECK_EQUAL(parse_string<long>("-1"), -1);
  BOOST_CHECK_EQUAL(parse_string<long>(" -1"), -1);
  BOOST_CHECK_EQUAL(parse_string<unsigned long>("10000000000"), 10000000000);
  BOOST_CHECK_EQUAL(parse_string<double>("1.0"), 1.0);
  BOOST_CHECK_EQUAL(parse_string<float>("1.0"), 1.0);
  BOOST_CHECK(boost::math::isnan(parse_string<double>("nan")));
  BOOST_CHECK_THROW(parse_string<long>("a"), std::invalid_argument);
  BOOST_CHECK_THROW(parse_string<long>("1 "), std::invalid_argument);
  BOOST_CHECK_THROW(parse_string<long>(""), std::invalid_argument);
  BOOST_CHECK_THROW(parse_string<long>(std::string("")), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_join) {
  std::vector<int> empty;
  boost::array<int,1> one = { 1 };
  boost::array<int,2> two = { -1, 1 };
  boost::array<double,3> three = { 0, 0.5, 1 };

  BOOST_CHECK_EQUAL(string_join(" ", empty), "");
  BOOST_CHECK_EQUAL(string_join(" ", one), "1");
  BOOST_CHECK_EQUAL(string_join(" ", two), "-1 1");
  BOOST_CHECK_EQUAL(string_join(",", two), "-1,1");
  BOOST_CHECK_EQUAL(string_join(", ", three), "0, 0.5, 1");
}

BOOST_AUTO_TEST_CASE(test_split) {
  std::vector<std::string> tokens;
  string_split("  23\ta4 5 ", " \t", tokens);
  BOOST_CHECK_EQUAL(tokens.size(), 3);
  BOOST_CHECK_EQUAL(tokens[0], "23");
  BOOST_CHECK_EQUAL(tokens[1], "a4");
  BOOST_CHECK_EQUAL(tokens[2], "5");

  std::string str = "  23\ta4 5";
  std::vector<const char*> ctokens;
  string_split(str, " \t", ctokens);
  BOOST_CHECK_EQUAL(ctokens.size(), 3);
  BOOST_CHECK_EQUAL(strcmp(ctokens[0], "23"), 0);
  BOOST_CHECK_EQUAL(strcmp(ctokens[1], "a4"), 0);
  BOOST_CHECK_EQUAL(strcmp(ctokens[2], "5"), 0);
}

BOOST_AUTO_TEST_CASE(test_split_path) {
  BOOST_CHECK_EQUAL(split_directory_file("/var/log/erase/all.txt"),
                    string_pair("/var/log/erase", "all.txt"));
}
