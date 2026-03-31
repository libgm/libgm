#define BOOST_TEST_MODULE domain_index_operations
#include <boost/test/unit_test.hpp>

#include <libgm/argument/named_argument.hpp>
#include <libgm/datastructure/domain_index.hpp>
#include <libgm/datastructure/indexed_domain.hpp>
#include <libgm/datastructure/domain_index_operations.hpp>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

using namespace libgm;

namespace {

constexpr std::size_t n = 100;

using Arg = NamedArg<16>;

struct Item {
  int id;
  IndexedDomain<Item, Arg> index;

  Item(int id, libgm::Domain<Arg> args)
    : id(id)
    , index(std::move(args)) {
    index.owner = this;
  }

  const libgm::Domain<Arg>& domain() const {
    return index.domain();
  }
};

libgm::Domain<Arg> make_random_domain(std::mt19937& rng, const std::vector<Arg>& pool) {
  libgm::Domain<Arg> domain;
  size_t size = 1 + (rng() % 10);
  domain.reserve(size);
  for (size_t j = 0; j < size; ++j) {
    domain.push_back(pool[rng() % pool.size()]);
  }
  domain.unique(); // Sort + deduplicate; operations assume ordered domains.
  return domain;
}

struct fixture {
  DomainIndex<Item, Arg> index;
  std::vector<std::unique_ptr<Item>> items;
  std::vector<libgm::Domain<Arg>> domains;

  fixture() {
    std::mt19937 rng(13);
    std::vector<Arg> pool;
    for (int i = 0; i < 20; ++i) {
      pool.emplace_back("a" + std::to_string(i));
    }

    items.reserve(n);
    domains.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      libgm::Domain<Arg> domain = make_random_domain(rng, pool);
      auto item = std::make_unique<Item>(static_cast<int>(i), domain);
      index.insert(item->index);
      domains.push_back(domain);
      items.push_back(std::move(item));
    }
  }
};

} // namespace

BOOST_FIXTURE_TEST_CASE(test_visit_covers, fixture) {
  for (size_t i = 0; i < n; ++i) {
    const libgm::Domain<Arg>& query = domains[i];
    int found = 0;

    visit_covers(index, query, [&](Item* item) {
      BOOST_CHECK(is_subset(query, item->domain()));
      ++found;
    });

    for (size_t j = 0; j < n; ++j) {
      if (is_subset(query, domains[j])) {
        --found;
      }
    }
    BOOST_CHECK_EQUAL(found, 0);
  }
}

BOOST_FIXTURE_TEST_CASE(test_visit_intersections, fixture) {
  for (size_t i = 0; i < n; ++i) {
    const libgm::Domain<Arg>& query = domains[i];
    int found = 0;

    visit_intersections(index, query, [&](Item* item) {
      BOOST_CHECK(!are_disjoint(query, item->domain()));
      ++found;
    });

    for (size_t j = 0; j < n; ++j) {
      if (!are_disjoint(query, domains[j])) {
        --found;
      }
    }
    BOOST_CHECK_EQUAL(found, 0);
  }
}

BOOST_FIXTURE_TEST_CASE(test_find_max_intersection, fixture) {
  for (size_t i = 0; i < n; ++i) {
    const libgm::Domain<Arg>& query = domains[i];
    Item* result = find_max_intersection(index, query);
    BOOST_REQUIRE(result != nullptr);

    size_t expected_max_meet = 0;
    size_t expected_min_size = std::numeric_limits<size_t>::max();
    for (size_t j = 0; j < n; ++j) {
      size_t meet = intersection_size(domains[j], query);
      if (meet > expected_max_meet) {
        expected_max_meet = meet;
        expected_min_size = domains[j].size();
      } else if (meet == expected_max_meet) {
        expected_min_size = std::min(expected_min_size, domains[j].size());
      }
    }

    size_t result_meet = intersection_size(result->domain(), query);
    BOOST_CHECK_EQUAL(result_meet, expected_max_meet);
    BOOST_CHECK_EQUAL(result->domain().size(), expected_min_size);
  }

  libgm::Domain<Arg> query = {Arg("__missing_arg__")};
  BOOST_CHECK(find_max_intersection(index, query) == nullptr);
}

BOOST_FIXTURE_TEST_CASE(test_find_min_cover, fixture) {
  for (size_t i = 0; i < n; ++i) {
    const libgm::Domain<Arg>& query = domains[i];
    Item* result = find_min_cover(index, query);
    BOOST_REQUIRE(result != nullptr);
    BOOST_CHECK(is_subset(query, result->domain()));

    size_t expected_min_size = std::numeric_limits<size_t>::max();
    for (size_t j = 0; j < n; ++j) {
      if (is_subset(query, domains[j])) {
        expected_min_size = std::min(expected_min_size, domains[j].size());
      }
    }
    BOOST_CHECK_EQUAL(result->domain().size(), expected_min_size);
  }

  libgm::Domain<Arg> query = {Arg("__missing_arg__")};
  BOOST_CHECK(find_min_cover(index, query) == nullptr);
}

BOOST_FIXTURE_TEST_CASE(test_is_maximal, fixture) {
  const libgm::Domain<Arg>& d0 = domains.front();
  libgm::Domain<Arg> covered_query = {d0.front()};
  BOOST_CHECK(!is_maximal(index, covered_query));

  Arg missing("__missing_arg__");
  libgm::Domain<Arg> uncovered_query = {missing};
  BOOST_CHECK(is_maximal(index, uncovered_query));

  libgm::Domain<Arg> mixed_query = covered_query;
  mixed_query.push_back(missing);
  mixed_query.sort();
  BOOST_CHECK(is_maximal(index, mixed_query));
}
