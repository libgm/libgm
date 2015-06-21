#define BOOST_TEST_MODULE text_dataset_format
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/text_dataset_format.hpp>

#include <fstream>

#include <boost/array.hpp>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_load_save) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  text_dataset_format format;
  universe u;
  format.load_config(dir + "/symbolic.cfg", u);

  BOOST_CHECK_EQUAL(format.separator, "\t");
  BOOST_CHECK_EQUAL(format.skip_rows, 0);
  BOOST_CHECK_EQUAL(format.skip_cols, 1);
  BOOST_CHECK_EQUAL(format.weighted, true);

  BOOST_CHECK_EQUAL(format.variables.size(), 3);

  BOOST_CHECK_EQUAL(format.variables[0].name(), "plain_finite");
  BOOST_CHECK_EQUAL(format.variables[0].levels().size(), 0);
  BOOST_CHECK_EQUAL(format.variables[0].num_values(), 4);
  BOOST_CHECK_EQUAL(format.variables[0].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.variables[1].name(), "named_finite");
  BOOST_CHECK_EQUAL(format.variables[1].levels().size(), 3);
  BOOST_CHECK_EQUAL(format.variables[1].num_values(), 3);
  BOOST_CHECK_EQUAL(format.variables[1].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.variables[2].name(), "vector");
  BOOST_CHECK_EQUAL(format.variables[2].num_dimensions(), 2);
  BOOST_CHECK_EQUAL(format.variables[2].is_continuous(), true);

  const char* filename2 = "text_dataset_format_tmp.cfg";
  format.save_config(filename2);

  boost::array<const char*, 11> lines = {
    "[options]",
    "separator=\"\\t\"",
    "missing=\"\"",
    "skip_rows=0",
    "skip_cols=1",
    "weighted=1",
    "",
    "[variables]",
    "plain_finite=finite(4)",
    "named_finite=a,b,c",
    "vector=vector(2)"
  };

  std::ifstream in(filename2);
  std::string line;
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(std::getline(in, line));
    BOOST_CHECK_EQUAL(line, lines[i]);
  }
}