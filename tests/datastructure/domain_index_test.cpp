#define BOOST_TEST_MODULE domain_index
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/datastructure/domain_index.hpp>
#include <libgm/datastructure/indexed_domain.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace libgm;

namespace {

using Arg = NamedArg<16>;

struct Item {
  int id;
  IndexedDomain<Item, Arg> index;

  Item(int id, libgm::Domain<Arg> args)
    : id(id)
    , index(std::move(args)) {
    index.owner = this;
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(test_domain_index_class_basics) {
  Arg a("cls_a");
  Arg b("cls_b");
  Arg c("cls_c");
  Arg d("cls_d");

  Item x(1, libgm::Domain<Arg>({a, b}));
  Item y(2, libgm::Domain<Arg>({b, c}));
  Item z(3, libgm::Domain<Arg>({c}));

  DomainIndex<Item, Arg> index;
  index.insert(x.index);
  index.insert(y.index);
  index.insert(z.index);

  BOOST_CHECK_EQUAL(index.count(a), 1);
  BOOST_CHECK_EQUAL(index.count(b), 2);
  BOOST_CHECK_EQUAL(index.count(c), 2);
  BOOST_CHECK_EQUAL(index.count(d), 0);

  BOOST_CHECK(index[a] == &x);
  BOOST_CHECK(index[c] == &y || index[c] == &z);
  BOOST_CHECK_THROW(index[d], std::out_of_range);

  {
    std::vector<int> ids;
    for (const IndexedDomain<Item, Arg>* indexed : index.adjacency(b)) {
      ids.push_back(indexed->item()->id);
    }
    std::sort(ids.begin(), ids.end());
    BOOST_CHECK(ids == std::vector<int>({1, 2}));
  }
  BOOST_CHECK(index.adjacency(d).empty());

  {
    libgm::Domain<Arg> args(index.arguments());
    args.sort();
    BOOST_CHECK(args == libgm::Domain<Arg>({a, b, c}));
  }

  index.erase(y.index);
  BOOST_CHECK_EQUAL(index.count(b), 1);
  BOOST_CHECK_EQUAL(index.count(c), 1);

  {
    std::vector<int> ids;
    for (const IndexedDomain<Item, Arg>* indexed : index.adjacency(b)) {
      ids.push_back(indexed->item()->id);
    }
    BOOST_CHECK(ids == std::vector<int>({1}));
  }

  index.clear();
  BOOST_CHECK(index.arguments().empty());
  BOOST_CHECK_EQUAL(index.count(a), 0);
  BOOST_CHECK(index.adjacency(a).empty());
}

BOOST_AUTO_TEST_CASE(test_domain_index_swap_and_empty_domain_insert) {
  Arg a("swap_a");
  Arg b("swap_b");
  Arg c("swap_c");

  Item x(10, libgm::Domain<Arg>({a, b}));
  Item y(20, libgm::Domain<Arg>({c}));
  Item e(30, libgm::Domain<Arg>());

  DomainIndex<Item, Arg> left;
  DomainIndex<Item, Arg> right;
  left.insert(x.index);
  right.insert(y.index);
  left.insert(e.index); // Empty domain: no adjacency entries.

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
