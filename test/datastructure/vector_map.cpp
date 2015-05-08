#define BOOST_TEST_MODULE vector_map
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/vector_map.hpp>

#include <map>

namespace libgm {
  template class vector_map<std::string, std::size_t>;
  template class vector_map<std::size_t, double>;
}

using namespace libgm;

typedef vector_map<std::string, std::size_t> map_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  map_type a;
  BOOST_CHECK(a.empty());

  map_type b = { {"x", 1}, {"y", 3} };
  BOOST_CHECK(!b.empty());
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b.count("x"), 1);
  BOOST_CHECK_EQUAL(b.count("y"), 1);
  BOOST_CHECK_EQUAL(b.count("z"), 0);

  map_type c(b.begin(), b.end());
  BOOST_CHECK(!c.empty());
  BOOST_CHECK_EQUAL(c.size(), 2);
  BOOST_CHECK_EQUAL(b, c);

  swap(a, c);
  BOOST_CHECK_EQUAL(c.size(), 0);
  BOOST_CHECK_EQUAL(a.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_accessors) {
  map_type x = { {"b", 2}, {"e", 5}, {"a", 1}, {"c", 3} };
  const map_type& cx = x;
  BOOST_CHECK_EQUAL(x.at("e"), 5);
  BOOST_CHECK_THROW(x.at("z"), std::out_of_range);

  BOOST_CHECK_EQUAL(cx.at("e"), 5);
  BOOST_CHECK_THROW(cx.at("z"), std::out_of_range);

  BOOST_CHECK_EQUAL(x.count("e"), 1);
  BOOST_CHECK_EQUAL(x.count("z"), 0);

  BOOST_CHECK(x.find("h") == x.end());
  BOOST_CHECK(x.find("a") == x.begin());

  BOOST_CHECK(x.equal_range("a") == std::make_pair(x.begin(), x.begin()+1));
  BOOST_CHECK(x.equal_range("e") == std::make_pair(x.end()-1, x.end()));
  BOOST_CHECK(x.equal_range("z") == std::make_pair(x.end(), x.end()));
  BOOST_CHECK(x.lower_bound("a") == x.begin());
  BOOST_CHECK(x.lower_bound("d") == x.end()-1);
  BOOST_CHECK(x.upper_bound("d") == x.end()-1);
  BOOST_CHECK(x.upper_bound("e") == x.end());
}

BOOST_AUTO_TEST_CASE(test_modifiers) {
  map_type x = { {"b", 2}, {"e", 5}, {"a", 1}, {"c", 3} };
  BOOST_CHECK(x.sorted());

  x.insert(std::make_pair("f", 6));
  BOOST_CHECK(x.sorted());

  x.insert(std::make_pair("d", 4));
  BOOST_CHECK(!x.sorted());
  x.sort();
  BOOST_CHECK(x.sorted());

  x.emplace("h", 8);
  BOOST_CHECK(x.sorted());
  BOOST_CHECK_EQUAL(x.at("h"), 8);

  map_type y = { {"b", 2}, {"e", 5} };
  map_type ys = { {"two", 2}, {"five", 5} };
  std::map<std::string, std::string> map = { {"b", "two"}, {"e", "five"} };
  y.subst_keys(map);
  BOOST_CHECK_EQUAL(y, ys);
}
