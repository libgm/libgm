#pragma once

#include "domain_index.hpp"

namespace libgm {

template <typename T, typename VISITOR>
void visit_covers(const DomainIndex<T>& index, const Domain& args, VISITOR visitor) {
  assert(!args.empty());

  // pick one value and iterate over all domains that contain that value
  for (T* item : index.adjacency(args.front())) {
    if (is_subset(args, item->domain())) {
      visitor(item);
    }
  }
}

template <typename T, typename VISITOR>
void visit_intersections(const DomainIndex<T>& index, const Domain& args, VISITOR visitor) {
  ankerl::unordered_dense::set<T*> result;
  for (Arg arg : args) {
    const auto& items = index.adjacency(arg);
    result.insert(items.begin(), items.end());
  }

  for (T* item : result) {
    visitor(item);
  }
}

template <typename T>
T* find_max_intersection(const DomainIndex<T>& index, const Domain& args) {
  assert(!args.empty());

  // A marker of all previously seen items.
  ankerl::unordered_dense::set<T*> visited;

  // determine the maximum intersection
  T* result = nullptr;
  size_t max_meet = 0;
  size_t min_size = std::numeric_limits<size_t>::max();
  for (Arg arg : args) {
    for (T* item : index.adjacency(arg)) {
      if (visited.contains(item)) continue;
      visited.insert(item);

      size_t intersection = intersection_size(item->domain(), args);
      if (intersection > max_meet ||
          (intersection == max_meet && item->domain().size() < min_size)) {
        max_meet = intersection;
        min_size = item->domain().size();
        result = item;
      }
    }
  }

  return result;
}

template <typename T>
T* find_min_cover(const DomainIndex<T>& index, const Domain& args) {
  assert(!args.empty());

  size_t min_size = std::numeric_limits<size_t>::max();
  T* result = nullptr;
  for (T* item : index.adjacency(args.front())) {
    if (item->domain().size() < min_size && is_subset(args, item->domain())) {
      min_size = item->domain().size();
      result = item;
    }
  }
  return result;
}

template <typename T>
bool is_maximal(const DomainIndex<T>& index, const Domain& args) {
  assert(!args.empty());

  // pick one argument and iterate over all sets that contain that argument
  for (T* item : index.adjacency(args.front())) {
    if (is_subset(args, item->domain())) {
      return false;
    }
  }
  return true;
}

} // namespace libgm
