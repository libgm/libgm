#define BOOST_TEST_MODULE mutable_queue
#include <boost/test/unit_test.hpp>

#include <queue>
#include <boost/random/mersenne_twister.hpp>
#include <libgm/datastructure/mutable_queue.hpp>

boost::mt19937 rng;
int n = 1000;

// Insert the same keys into STL queues and LibGM queues and make sure
// they spit out the keys in the same order.
BOOST_AUTO_TEST_CASE(test_insert) {
  std::priority_queue<int> std_pq;
  libgm::mutable_queue<int, int> prl_pq;
  for (int i = 0; i < n; i++) {
    int x = rng();
    int y = rng();
    prl_pq.push(x, y);
    std_pq.push(y);
    BOOST_CHECK_EQUAL(std_pq.size(), prl_pq.size());
  }
  while (!std_pq.empty()) {
    assert(!prl_pq.empty());
    std::pair<int,int> xy;
    int std_y = std_pq.top();
    xy = prl_pq.pop();
    BOOST_CHECK_EQUAL(xy.second, std_y);
    std_pq.pop();
  }
}

// Insert a bunch of items into the LibGM queue, reprioritize them,
// and then make sure they come out in sorted order.
BOOST_AUTO_TEST_CASE(test_reprioritize) {
  libgm::mutable_queue<int, int> prl_pq;
  for (int i = 0; i < n; i++) {
    int x = rng();
    prl_pq.push(i, x);
  }
  for (int i = 0; i < n; i++) {
    int x = rng();
    prl_pq.update(i, x);
  }
  int x = prl_pq.pop().second;
  while (!prl_pq.empty()) {
    int y = prl_pq.pop().second;
    BOOST_CHECK_LE(y, x);
    x = y;
  }
}
