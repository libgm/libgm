#pragma once

#include <iterator>

namespace libgm {

/**
 * An output iterator that counts how many times a value has been stored.
 * Similarly to insert iterators in STL, operator* returns itself, so that
 * operator= can increase the counter.
 * \ingroup iterator
 */
class CountingOutputIterator
  : public std::iterator<std::output_iterator_tag, void, void, void, void> {

public:
  /// Constructor.
  CountingOutputIterator() : counter_() {}

  /// Increment the counter.
  template <typename T>
  CountingOutputIterator& operator=(const T&) {
    ++counter_;
    return *this;
  }

  /// Dereferences to en empty target object that can absorb any assignment.
  CountingOutputIterator& operator*() {
    return *this;
  }

  /// Returns *this with no side-effects.
  CountingOutputIterator& operator++() {
    return *this;
  }

  /// Returns *this with no side-effects.
  CountingOutputIterator& operator++(int) {
    return *this;
  }

  /// Returns the number of positions that have been assigned.
  size_t count() const {
    return counter_;
  }

private:
  /// The counter.
  size_t counter_;

}; // class CountingOutputIteraotr

} // namespace libgm
