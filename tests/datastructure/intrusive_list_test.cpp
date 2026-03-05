#define BOOST_TEST_MODULE intrusive_list
#include <boost/test/unit_test.hpp>

#include <libgm/datastructure/intrusive_list.hpp>

#include <memory>
#include <vector>

using namespace libgm;

namespace {

struct Item {
  int id;
  IntrusiveList<Item>::Hook hook_a;
  IntrusiveList<Item>::Hook hook_b;
};

std::vector<int> ids(const IntrusiveList<Item>& list) {
  std::vector<int> result;
  for (const Item* item : list) {
    result.push_back(item->id);
  }
  return result;
}

} // namespace

BOOST_AUTO_TEST_CASE(test_push_and_iteration_order) {
  IntrusiveList<Item> list;
  Item x{1}, y{2}, z{3};

  BOOST_CHECK(list.empty());

  list.push_back(&x, x.hook_a);
  list.push_back(&y, y.hook_a);
  list.push_back(&z, z.hook_a);

  BOOST_CHECK(!list.empty());
  BOOST_CHECK_EQUAL(list.front()->id, 1);
  BOOST_CHECK_EQUAL(list.back()->id, 3);
  BOOST_CHECK(ids(list) == std::vector<int>({1, 2, 3}));
}

BOOST_AUTO_TEST_CASE(test_push_front_and_erase_resets_hook) {
  IntrusiveList<Item> list;
  Item x{1}, y{2}, z{3};

  list.push_front(&y, y.hook_a); // [2]
  list.push_front(&x, x.hook_a); // [1, 2]
  list.push_back(&z, z.hook_a);  // [1, 2, 3]
  BOOST_CHECK(ids(list) == std::vector<int>({1, 2, 3}));

  list.erase(&y, y.hook_a);      // [1, 3]
  BOOST_CHECK(ids(list) == std::vector<int>({1, 3}));
  BOOST_CHECK(!y.hook_a);

  list.erase(&x, x.hook_a);      // [3]
  list.erase(&z, z.hook_a);      // []
  BOOST_CHECK(list.empty());
  BOOST_CHECK(!x.hook_a);
  BOOST_CHECK(!z.hook_a);
}

BOOST_AUTO_TEST_CASE(test_same_item_in_two_lists_with_two_hooks) {
  IntrusiveList<Item> a_list;
  IntrusiveList<Item> b_list;
  Item x{1}, y{2};

  a_list.push_back(&x, x.hook_a);
  a_list.push_back(&y, y.hook_a);
  b_list.push_back(&y, y.hook_b);
  b_list.push_back(&x, x.hook_b);

  BOOST_CHECK(ids(a_list) == std::vector<int>({1, 2}));
  BOOST_CHECK(ids(b_list) == std::vector<int>({2, 1}));

  a_list.erase(&y, y.hook_a);
  BOOST_CHECK(ids(a_list) == std::vector<int>({1}));
  BOOST_CHECK(ids(b_list) == std::vector<int>({2, 1}));
  BOOST_CHECK(!y.hook_a);
  BOOST_CHECK(y.hook_b);
}

BOOST_AUTO_TEST_CASE(test_hook_destructor_unlinks_automatically) {
  IntrusiveList<Item> list;
  auto x = std::make_unique<Item>(Item{1});
  auto y = std::make_unique<Item>(Item{2});

  list.push_back(x.get(), x->hook_a);
  list.push_back(y.get(), y->hook_a);
  BOOST_CHECK(ids(list) == std::vector<int>({1, 2}));

  x.reset(); // Hook destructor should unlink from list.
  BOOST_CHECK(ids(list) == std::vector<int>({2}));

  y.reset(); // Last item unlinked, list becomes empty.
  BOOST_CHECK(list.empty());
}

BOOST_AUTO_TEST_CASE(test_list_destructor_clears_all_hooks) {
  Item x{1}, y{2}, z{3};
  {
    IntrusiveList<Item> list;
    list.push_back(&x, x.hook_a);
    list.push_back(&y, y.hook_a);
    list.push_back(&z, z.hook_a);
    BOOST_CHECK(x.hook_a);
    BOOST_CHECK(y.hook_a);
    BOOST_CHECK(z.hook_a);
  }

  BOOST_CHECK(!x.hook_a);
  BOOST_CHECK(!y.hook_a);
  BOOST_CHECK(!z.hook_a);
}

BOOST_AUTO_TEST_CASE(test_each_list_destruction_clears_its_own_hook) {
  Item x{1};
  {
    IntrusiveList<Item> list_a;
    {
      IntrusiveList<Item> list_b;
      list_a.push_back(&x, x.hook_a);
      list_b.push_back(&x, x.hook_b);
      BOOST_CHECK(x.hook_a);
      BOOST_CHECK(x.hook_b);
    }
    BOOST_CHECK(x.hook_a);
    BOOST_CHECK(!x.hook_b);
  }
  BOOST_CHECK(!x.hook_a);
  BOOST_CHECK(!x.hook_b);
}

BOOST_AUTO_TEST_CASE(test_move_constructor_transfers_links) {
  Item x{1}, y{2}, z{3};
  IntrusiveList<Item> source;
  source.push_back(&x, x.hook_a);
  source.push_back(&y, y.hook_a);
  source.push_back(&z, z.hook_a);

  IntrusiveList<Item> moved(std::move(source));

  BOOST_CHECK(source.empty());
  BOOST_CHECK(ids(moved) == std::vector<int>({1, 2, 3}));
  BOOST_CHECK(x.hook_a);
  BOOST_CHECK(y.hook_a);
  BOOST_CHECK(z.hook_a);
}
