#pragma once

#include <libgm/datastructure/intrusive_list.hpp>

#include <utility>

namespace libgm {

template <typename V, typename E>
class IntrusiveEdge {
public:
  struct Connectivity {
    V* vertex[2];
    typename IntrusiveList<E>::Hook adjacency_hook[2];
  };

  IntrusiveEdge() = default;

  IntrusiveEdge(typename IntrusiveList<E>::Entry entry)
    : entry_(entry) {}

  IntrusiveEdge(E* edge)
    : entry_{edge, nullptr} {
    if (edge) entry_.hook = connectivity().adjacency_hook;
  }

  operator std::pair<V*, V*>() const {
    return {source(), target()};
  }

  E* get() const {
    return entry_.item;
  }

  E* operator->() const {
    return entry_.item;
  }

  bool index() const {
    ptrdiff_t diff = entry_.hook - connectivity().adjacency_hook;
    assert((diff & ~ptrdiff_t(1)) == 0 && "Invalid hook");
    return bool(diff);
  }

  V* source() const {
    return connectivity().vertex[!index()];
  }

  V* target() const {
    return connectivity().vertex[index()];
  }

  IntrusiveEdge reverse() const {
    return IntrusiveEdge({entry_.item, connectivity().adjacency_hook + !index()});
  }

  bool operator==(const IntrusiveEdge& other) const {
    return entry_ == other.entry_;
  }

  bool operator!=(const IntrusiveEdge& other) const {
    return entry_ != other.entry_;
  }

private:
  Connectivity& connectivity() const {
    // This is guaranteed to work because E is standard layout
    return *reinterpret_cast<Connectivity*>(entry_.item);
  }

  typename IntrusiveList<E>::Entry entry_;
};

}
