#define BOOST_TEST_MODULE text_dataset_format
#include <boost/test/unit_test.hpp>

#include <libgm/learning/dataset/text_dataset_format.hpp>

#include <fstream>
#include <vector>

using namespace libgm;

BOOST_AUTO_TEST_CASE(test_load_save) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  text_dataset_format format;
  universe u;
  format.load_config(dir + "/text_format.cfg", u);

  BOOST_CHECK_EQUAL(format.separator, "\t");
  BOOST_CHECK_EQUAL(format.skip_rows, 0);
  BOOST_CHECK_EQUAL(format.skip_cols, 1);
  BOOST_CHECK_EQUAL(format.weighted, true);

  BOOST_CHECK_EQUAL(format.variables.size(), 3);

  BOOST_CHECK_EQUAL(format.variables[0].name(), "plain_discrete");
  BOOST_CHECK_EQUAL(format.variables[0].levels().size(), 0);
  BOOST_CHECK_EQUAL(format.variables[0].num_values(), 4);
  BOOST_CHECK_EQUAL(format.variables[0].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.variables[1].name(), "named_discrete");
  BOOST_CHECK_EQUAL(format.variables[1].levels().size(), 3);
  BOOST_CHECK_EQUAL(format.variables[1].num_values(), 3);
  BOOST_CHECK_EQUAL(format.variables[1].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.variables[2].name(), "vector");
  BOOST_CHECK_EQUAL(format.variables[2].num_dimensions(), 2);
  BOOST_CHECK_EQUAL(format.variables[2].is_continuous(), true);

  BOOST_CHECK_EQUAL(format.dprocesses[0].name(), "plain_discrete");
  BOOST_CHECK_EQUAL(format.dprocesses[0].levels().size(), 0);
  BOOST_CHECK_EQUAL(format.dprocesses[0].num_values(), 5);
  BOOST_CHECK_EQUAL(format.dprocesses[0].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.dprocesses[1].name(), "named_discrete");
  BOOST_CHECK_EQUAL(format.dprocesses[1].levels().size(), 4);
  BOOST_CHECK_EQUAL(format.dprocesses[1].num_values(), 4);
  BOOST_CHECK_EQUAL(format.dprocesses[1].is_discrete(), true);

  BOOST_CHECK_EQUAL(format.dprocesses[2].name(), "vector");
  BOOST_CHECK_EQUAL(format.dprocesses[2].num_dimensions(), 3);
  BOOST_CHECK_EQUAL(format.dprocesses[2].is_continuous(), true);

  const char* filename2 = "text_dataset_format_tmp.cfg";
  format.save_config(filename2);

  std::vector<const char*> lines = {
    "[discrete_processes]",
    "plain_discrete=discrete(5)",
    "named_discrete=a,b,c,d",
    "vector=continuous(3)",
    "",
    "[options]",
    "separator=\"\\t\"",
    "missing=\"\"",
    "skip_rows=0",
    "skip_cols=1",
    "weighted=1",
    "",
    "[variables]",
    "plain_discrete=discrete(4)",
    "named_discrete=a,b,c",
    "vector=continuous(2)"
  };

  std::ifstream in(filename2);
  std::string line;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    BOOST_CHECK(std::getline(in, line));
    BOOST_CHECK_EQUAL(line, lines[i]);
  }
}
