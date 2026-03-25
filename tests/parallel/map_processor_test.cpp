#define BOOST_TEST_MODULE map_processor
#include <boost/test/unit_test.hpp>

#include <ankerl/unordered_dense.h>

#include <libgm/parallel/map_processor.hpp>

#include <numeric>
#include <string>
#include <vector>

using namespace libgm;

BOOST_AUTO_TEST_CASE(processes_map_without_state) {
  ankerl::unordered_dense::map<std::string, int> jobs = {
    {"a", 1},
    {"bb", 2},
    {"ccc", 3}
  };

  MapProcessor<std::string, int> process([](const std::string& key, int& value) {
    value += static_cast<int>(key.size());
  });

  process(jobs, 2);

  BOOST_CHECK_EQUAL(jobs.at("a"), 2);
  BOOST_CHECK_EQUAL(jobs.at("bb"), 4);
  BOOST_CHECK_EQUAL(jobs.at("ccc"), 6);
}

BOOST_AUTO_TEST_CASE(processes_map_with_state) {
  ankerl::unordered_dense::map<int, int> jobs = {
    {1, 10},
    {2, 20},
    {3, 30},
    {4, 40}
  };

  std::vector<int> state(3, 0);
  MapProcessor<int, int, int> process([](const int& key, int& value, int& sum) {
    value += key;
    sum += value;
  });

  process(jobs, state);

  BOOST_CHECK_EQUAL(jobs.at(1), 11);
  BOOST_CHECK_EQUAL(jobs.at(2), 22);
  BOOST_CHECK_EQUAL(jobs.at(3), 33);
  BOOST_CHECK_EQUAL(jobs.at(4), 44);
  BOOST_CHECK_EQUAL(std::accumulate(state.begin(), state.end(), 0), 110);
}
