#pragma once

#include <cassert>
#include <vector>

#include <ankerl/unordered_dense.h>

namespace libgm {

/**
 * A heap implementation of a priority queue that supports external
 * priorities and priority updates. Both template arguments must be
 * Assignable, EqualityComparable, and LessThanComparable.
 *
 * @param T
 *        the type of items stored in the priority queue.
 * @param Priority
 *        the type used to prioritize items.
 *
 * \ingroup datastructure
 */
template <typename T, typename Priority>
class MutableQueue {
public:
  /// An element of the heap.
  using value_type = typename std::pair<T, Priority>;

  /// Default constructor.
  MutableQueue()
    : heap_(1, std::make_pair(T(), Priority())) { }

  /// Returns the number of elements in the heap.
  size_t size() const {
    return heap_.size() - 1;
  }

  /// Returns true iff the queue is empty.
  bool empty() const {
    return size() == 0;
  }

  /// Returns true if the queue contains the given value
  bool contains(const T& item) const {
    return index_map_.count(item) > 0;
  }

  /// Enqueues a new item in the queue.
  void push(const T& item, Priority priority) {
    heap_.push_back(std::make_pair(item, priority));
    size_t i = size();
    index_map_[item] = i;
    while ((i > 1) && (priority_at(parent(i)) < priority)) {
      swap(i, parent(i));
      i = parent(i);
    }
  }

  /// Accesses the item with maximum priority in the queue.
  const std::pair<T, Priority>& top() const {
    return heap_[1];
  }

  /**
   * Removes the item with maximum priority from the queue, and
   * returns it with its priority.
   */
  std::pair<T, Priority> pop() {
    value_type top = heap_[1];
    if (size() != 1)
      swap(1, size());
    heap_.pop_back();
    heapify(1);
    index_map_.erase(top.first);
    return top;
  }

  /// Returns the weight associated with a key
  Priority get(const T& item) const {
    return heap_[index_map_.at(item)].second;
  }

  /// Returns the priority associated with a key
  Priority operator[](const T& item) const {
    return get(item);
  }

  /**
   * Updates the priority associated with a item in the queue. This
   * function fails if the item is not already present.
  */
  void update(const T& item, Priority priority) {
    // Verify that the item is currently in the queue
    auto it = index_map_.find(item);
    assert(it != index_map_.end());
    // If it is already present update the priority
    size_t i = it->second;
    heap_[i].second = priority;
    while ((i > 1) && (priority_at(parent(i)) < priority)) {
      swap(i, parent(i));
      i = parent(i);
    }
    heapify(i);
  }

  /**
   * If item is already in the queue, sets its priority to the maximum
   * of the old priority and the new one. If the item is not in the queue,
   * adds it to the queue.
   */
  void updating_insert_max(const T& item, Priority priority) {
    if(!contains(item))
      push(item, priority);
    else{
      Priority effective_priority = std::max(get(item), priority);
      update(item, effective_priority);
    }
  }

  /**
   * If item is already in the queue, sets its priority to the sum
   * of the old priority and the new one.
   * If the item is not in the queue, adds it to the queue.
   */
  void updating_insert_sum(const T& item, Priority priority) {
    if(!contains(item))
      push(item, priority);
    else {
      Priority effective_priority = get(item) + priority;
      update(item, effective_priority);
    }
  }

  /// Returns the values (key-priority pairs) in the priority queue
  const std::vector<value_type>& values() const {
    return heap_;
  }

  /// Clears all the values (equivalent to stl clear)
  void clear() {
    heap_.clear();
    heap_.push_back(std::make_pair(T(), Priority()));
    index_map_.clear();
  }

  /// Remove an item from the queue.
  /// Note: The item MUST be in the queue.
  void remove(const T& item) {
    // Ensure that the element is in the queue
    auto it = index_map_.find(item);
    assert(it != index_map_.end());
    size_t i = it->second;
    swap(i, size());
    heap_.pop_back();
    heapify(i);
    // erase the element from the index map
    index_map_.erase(it);
  }

  /// Remove an item from the queue if it is present.
  void remove_if_present(const T& item) {
    auto it = index_map_.find(item);
    if (it == index_map_.end())
      return;
    size_t i = it->second;
    swap(i, size());
    heap_.pop_back();
    heapify(i);
    // erase the element from the index map
    index_map_.erase(it);
  }

  /// Increment the priority of an item in the queue by inc if it is present.
  /// @return  True iff the item was present.
  bool increment_if_present(const T& item, Priority inc) {
    auto it = index_map_.find(item);
    if (it == index_map_.end())
      return false;
    size_t i = it->second;
    Priority new_priority = heap_[i].second + inc;
    heap_[i].second = new_priority;
    while ((i > 1) && (priority_at(parent(i)) < new_priority)) {
      swap(i, parent(i));
      i = parent(i);
    }
    heapify(i);
    return true;
  }

private:
  /// Returns the index of the left child of the supplied index.
  static size_t left(size_t i) {
    return 2 * i;
  }

  /// Returns the index of the right child of the supplied index.
  static size_t right(size_t i) {
    return 2 * i + 1;
  }

  /// Returns the index of the parent of the supplied index.
  static size_t parent(size_t i) {
    return i / 2;
  }

  /// Extracts the priority at a heap location.
  Priority priority_at(size_t i) {
    return heap_[i].second;
  }

  /// Compares the priorities at two heap locations.
  bool less(size_t i, size_t j) {
    return heap_[i].second < heap_[j].second;
  }

  /// Swaps the heap locations of two elements.
  void swap(size_t i, size_t j) {
    std::swap(heap_[i], heap_[j]);
    index_map_[heap_[i].first] = i;
    index_map_[heap_[j].first] = j;
  }

  /// The traditional heapify function.
  void heapify(size_t i) {
    size_t l = left(i);
    size_t r = right(i);
    size_t s = size();
    size_t largest = i;
    if ((l <= s) && less(i, l))
      largest = l;
    if ((r <= s) && less(largest, r))
      largest = r;
    if (largest != i) {
      swap(i, largest);
      heapify(largest);
    }
  }

  /// The heap used to store the elements. The first element is unused.
  std::vector<value_type> heap_;

  /// The map used to map from items to indexes in the heap.
  ankerl::unordered_dense::map<T, size_t> index_map_;
}; // class MutableQueue

}