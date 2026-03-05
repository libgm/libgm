#define BOOST_TEST_MODULE domain_index
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/datastructure/domain_index.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace libgm;

namespace {

struct Item {
  int id;
  Domain args;
  IntrusiveList<Item>::HookArray index_hooks;

  Item(int id, Domain args)
    : id(id)
    , args(std::move(args))
    , index_hooks(this->args.size()) {}

  const Domain& domain() const {
    return args;
  }
};

Arg make_arg(const std::string& name) {
  return NamedFactory::default_factory().make(name);
}

} // namespace

BOOST_AUTO_TEST_CASE(test_domain_index_class_basics) {
  Arg a = make_arg("cls_a");
  Arg b = make_arg("cls_b");
  Arg c = make_arg("cls_c");
  Arg d = make_arg("cls_d");

  Item x(1, Domain({a, b}));
  Item y(2, Domain({b, c}));
  Item z(3, Domain({c}));

  DomainIndex<Item> index;
  index.insert(&x, x.index_hooks);
  index.insert(&y, y.index_hooks);
  index.insert(&z, z.index_hooks);

  BOOST_CHECK_EQUAL(index.count(a), 1);
  BOOST_CHECK_EQUAL(index.count(b), 2);
  BOOST_CHECK_EQUAL(index.count(c), 2);
  BOOST_CHECK_EQUAL(index.count(d), 0);

  BOOST_CHECK(index[a] == &x);
  BOOST_CHECK(index[c] == &y || index[c] == &z);
  BOOST_CHECK_THROW(index[d], std::out_of_range);

  {
    std::vector<int> ids;
    for (const Item* item : index.adjacency(b)) {
      ids.push_back(item->id);
    }
    std::sort(ids.begin(), ids.end());
    BOOST_CHECK(ids == std::vector<int>({1, 2}));
  }
  BOOST_CHECK(index.adjacency(d).empty());

  {
    Domain args(index.arguments());
    args.sort();
    BOOST_CHECK(args == Domain({a, b, c}));
  }

  index.erase(&y, y.index_hooks);
  BOOST_CHECK_EQUAL(index.count(b), 1);
  BOOST_CHECK_EQUAL(index.count(c), 1);

  {
    std::vector<int> ids;
    for (const Item* item : index.adjacency(b)) {
      ids.push_back(item->id);
    }
    BOOST_CHECK(ids == std::vector<int>({1}));
  }

  index.clear();
  BOOST_CHECK(index.arguments().empty());
  BOOST_CHECK_EQUAL(index.count(a), 0);
  BOOST_CHECK(index.adjacency(a).empty());
}

BOOST_AUTO_TEST_CASE(test_domain_index_swap_and_empty_domain_insert) {
  Arg a = make_arg("swap_a");
  Arg b = make_arg("swap_b");
  Arg c = make_arg("swap_c");

  Item x(10, Domain({a, b}));
  Item y(20, Domain({c}));
  Item e(30, Domain());

  DomainIndex<Item> left;
  DomainIndex<Item> right;
  left.insert(&x, x.index_hooks);
  right.insert(&y, y.index_hooks);
  left.insert(&e, e.index_hooks); // Empty domain: no adjacency entries.

  BOOST_CHECK_EQUAL(left.count(a), 1);
  BOOST_CHECK_EQUAL(right.count(c), 1);

  swap(left, right);

  BOOST_CHECK_EQUAL(left.count(c), 1);
  BOOST_CHECK_EQUAL(left.count(a), 0);
  BOOST_CHECK(left[c] == &y);

  BOOST_CHECK_EQUAL(right.count(a), 1);
  BOOST_CHECK_EQUAL(right.count(c), 0);
  BOOST_CHECK(right[a] == &x);
}
