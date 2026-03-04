#define BOOST_TEST_MODULE set_index
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/set_index.hpp>
#include <libgm/iterator/counting_output_iterator.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace libgm {
  template class set_index<int, std::vector<double>>;
}

using namespace libgm;
std::size_t n = 200;

struct fixture {
  typedef std::list<int> list_type;
  typedef std::set<int> set_type;

  fixture() {
    std::mt19937 rng;
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t size = 1 + (rng() % 10);
      list_type list;
      for (std::size_t j = 0; j < size; j++) {
        list.push_back(rng() % 20);
      }
      index.insert(int(i), list);
      lists.push_back(list);
      sets.emplace_back(list.begin(), list.end());
    }
  }

  bool disjoint(const set_type& a, const set_type& b) {
    counting_output_iterator out;
    return std::set_intersection(a.begin(), a.end(),
                                 b.begin(), b.end(), out).count() == 0;
  }

  set_index<int, list_type> index;
  std::vector<list_type> lists;
  std::vector<set_type> sets;
};

BOOST_FIXTURE_TEST_CASE(test_superset, fixture) {
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<int> results;
    int found = 0;
    index.supersets(lists[i], [&](int j) {
        BOOST_CHECK(std::includes(sets[j].begin(), sets[j].end(),
                                  sets[i].begin(), sets[i].end()));
        ++found;
      });

    // check the answers are complete
    for (int j = 0; j < n; ++j) {
      if (std::includes(sets[j].begin(), sets[j].end(),
                        sets[i].begin(), sets[i].end())) {
        --found;
      }
    }
    BOOST_CHECK_EQUAL(found, 0);
  }
}

BOOST_FIXTURE_TEST_CASE(test_intersection, fixture) {
  for (int i = 0; i < n; ++i) {
    std::vector<int> results;
    int found = 0;
    index.intersecting_sets(lists[i], [&](int j) {
        BOOST_CHECK(!disjoint(sets[j], sets[i]));
        ++found;
      });

    // Check the answers are complete.
    for (std::size_t j = 0; j < n; j++) {
      if (!disjoint(sets[j], sets[i])) {
        --found;
      }
    }
    BOOST_CHECK_EQUAL(found, 0);
  }
}
