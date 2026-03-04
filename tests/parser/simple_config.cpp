#define BOOST_TEST_MODULE simple_config
#include <boost/test/unit_test.hpp>

#include <libgm/parser/simple_config.hpp>

using namespace libgm;

BOOST_TEST_DONT_PRINT_LOG_VALUE(simple_config::config_entries);

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);

  char* filename = boost::unit_test::framework::master_test_suite().argv[1];
  simple_config config;
  config.load(filename);

  simple_config::config_entries section1;
  section1.push_back(std::make_pair("key1", "value1"));
  section1.push_back(std::make_pair("key2", "value2"));
  section1.push_back(std::make_pair("key3", ""));

  simple_config::config_entries section2;
  section2.push_back(std::make_pair("key4", "4"));

  simple_config::config_entries section3;

  BOOST_CHECK_EQUAL(config["section1"], section1);
  BOOST_CHECK_EQUAL(config["section2"], section2);
  BOOST_CHECK_EQUAL(config["section3"], section3);
}

BOOST_AUTO_TEST_CASE(test_save) {
  const char* filename = "simple_config_tmp.cfg";

  simple_config config;
  config.add("sectionA", "key_str", "some string");
  config.add("sectionA", "key_int", 2);
  config.add("sectionB", "key_bool", true);
  config.save(filename);

  std::vector<std::string> lines;
  lines.push_back("[sectionA]");
  lines.push_back("key_str=some string");
  lines.push_back("key_int=2");
  lines.push_back("[sectionB]");
  lines.push_back("key_bool=1");

  std::ifstream in(filename);
  std::string line;
  std::size_t i = 0;
  while (getline(in, line)) {
    if (!line.empty()) {
      if (i < lines.size()) {
        BOOST_CHECK_EQUAL(line, lines[i]);
      }
      ++i;
    }
  }

  BOOST_CHECK_EQUAL(i, lines.size());
}
