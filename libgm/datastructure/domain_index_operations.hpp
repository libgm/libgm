#pragma once

#include "domain_index.hpp"

namespace libgm {

template <typename Item, typename VISITOR>
void visit_covers(const DomainIndex<Item>& index, const Domain& args, VISITOR visitor) {
  assert(!args.empty());

  // pick one value and iterate over all domains that contain that value
  for (IndexedDomain<Item>* indexed : index.adjacency(args.front())) {
    if (is_subset(args, indexed->domain())) {
      visitor(indexed->item());
    }
  }
}

template <typename Item, typename VISITOR>
void visit_intersections(const DomainIndex<Item>& index, const Domain& args, VISITOR visitor) {
  ankerl::unordered_dense::set<Item*> result;
  for (Arg arg : args) {
    for (IndexedDomain<Item>* indexed : index.adjacency(arg)) {
      result.insert(indexed->item());
    }
  }

  for (Item* item : result) {
    visitor(item);
  }
}

template <typename Item>
Item* find_max_intersection(const DomainIndex<Item>& index, const Domain& args) {
  assert(!args.empty());

  // A marker of all previously seen items.
  ankerl::unordered_dense::set<Item*> visited;

  // determine the maximum intersection
  Item* result = nullptr;
  size_t max_meet = 0;
  size_t min_size = std::numeric_limits<size_t>::max();
  for (Arg arg : args) {
    for (IndexedDomain<Item>* indexed : index.adjacency(arg)) {
      Item* item = indexed->item();
      if (visited.contains(item)) continue;
      visited.insert(item);

      size_t intersection = intersection_size(indexed->domain(), args);
      if (intersection > max_meet ||
          (intersection == max_meet && indexed->domain().size() < min_size)) {
        max_meet = intersection;
        min_size = indexed->domain().size();
        result = item;
      }
    }
  }

  return result;
}

template <typename Item>
Item* find_min_cover(const DomainIndex<Item>& index, const Domain& args) {
  assert(!args.empty());

  size_t min_size = std::numeric_limits<size_t>::max();
  Item* result = nullptr;
  for (IndexedDomain<Item>* indexed : index.adjacency(args.front())) {
    if (indexed->domain().size() < min_size && is_subset(args, indexed->domain())) {
      min_size = indexed->domain().size();
      result = indexed->item();
    }
  }
  return result;
}

template <typename Item>
bool is_maximal(const DomainIndex<Item>& index, const Domain& args) {
  assert(!args.empty());

  // pick one argument and iterate over all sets that contain that argument
  for (IndexedDomain<Item>* indexed : index.adjacency(args.front())) {
    if (is_subset(args, indexed->domain())) {
      return false;
    }
  }
  return true;
}

} // namespace libgm
