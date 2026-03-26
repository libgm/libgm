#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/intrusive_list.hpp>
#include <libgm/datastructure/unordered_dense.hpp>
#include <libgm/iterator/map_key_iterator.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <ranges>
#include <vector>

namespace libgm {

/**
 * An index over domains that efficiently processes intersection and
 * superset queries. Each domain is associated with a pointer handle
 * that stores that domain.
 */
template <typename T>
class DomainIndex {
  // Public types
  //==========================================================================
public:
  /// Maps each element to the vector of containing pointers.
  using AdjacencyMap = ankerl::unordered_dense::map<Arg, IntrusiveList<T>>;

  /// A vector of intrusive list hooks.
  using HookArray = typename IntrusiveList<T>::HookArray;

  /// Iterators over arguments contained in this index.
  /// FIXME: eliminate dead arguments
  using argument_iterator = MapKeyIterator<AdjacencyMap>;

  // Constructors and destructors
  //==========================================================================
public:
  /// Creates an empty set index with the given domain and hash members
  DomainIndex() = default;

  /// Swaps the content of two index sets in constant time.
  friend void swap(DomainIndex& a, DomainIndex& b) {
    using std::swap;
    swap(a.adjacency_, b.adjacency_);
  }

  // Queries
  //==========================================================================

  /**
   * Returns the range of arguments in this index.
   */
  std::ranges::subrange<argument_iterator> arguments() const {
    return { adjacency_.begin(), adjacency_.end() };
  }

  /**
   * Returns the number of domains with the specified argument.
   */
  size_t count(Arg arg) const {
    const IntrusiveList<T>& domains = adjacency(arg);
    return std::distance(domains.begin(), domains.end());
  }

  /**
   * Returns the number of distinct arguments present in this index.
   */
  size_t num_arguments() const {
    return adjacency_.size();
  }

  /**
   * Returns a handle for any domain that contains the specified argument.
   * \return the handle or nullptr if no there is no domain containing the argument
   * \throw std::out_of_range if there is no such argument
   */
  T* operator[](Arg arg) const {
    return adjacency_.at(arg).front();
  }

  const IntrusiveList<T>& adjacency(Arg arg) const {
    static IntrusiveList<T> empty;
    auto it = adjacency_.find(arg);
    if (it == adjacency_.end()) {
      return empty;
    } else {
      return it->second;
    }
  }

  // Mutating operations
  //==========================================================================

  /**
   * Inserts a new domain in the index. This function is linear in the number of arguments of the
   * domain. The domain must not be empty, must not be present in the index.
   */
  void insert(T* item, HookArray& hooks) {
    const Domain& domain = item->domain();
    for (size_t i = 0; i < domain.size(); ++i) {
      adjacency_[domain[i]].push_back(item, hooks[i]);
    }
  }

  /// Removes the set with the given handle from the index.
  void erase(T* item, HookArray& hooks) {
    const Domain& domain = item->domain();
    for (size_t i = 0; i < domain.size(); ++i) {
      adjacency_[domain[i]].erase(item, hooks[i]);
    }
  }

  /// Removes all sets from this index.
  void clear() {
    adjacency_.clear();
  }

  /// Prints the index to an output stream
  friend std::ostream& operator<<(std::ostream& out, const DomainIndex& index) {
    for (const auto& [arg, items] : index.adjacency_) {
      out << arg << " -->";
      for (const T* item : items) {
        out << ' ' << item;
      }
      out << std::endl;
    }
    return out;
  }

  // Private members
  //==========================================================================
private:
  /// The mapping from arguments to handles (domains).
  AdjacencyMap adjacency_;

}; // class DomainIndex

} // namespace libgm
