#ifndef LIBGM_MUTABLE_PRIORITY_QUEUE_HPP
#define LIBGM_MUTABLE_PRIORITY_QUEUE_HPP

#include <cassert>
#include <map>
#include <vector>

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
   * @see Boost's mutable_queue in boost/pending/mutable_queue.hpp
   * @todo Add a comparator
   *
   * \ingroup datastructure
   */
  template <typename T, typename Priority>
  class mutable_queue {
  public:

    //! An element of the heap.
    typedef typename std::pair<T, Priority> heap_element;

  protected:

    //! The storage type of the index map
    typedef std::map<T, std::size_t> index_map_type;

    //! The heap used to store the elements. The first element is unused.
    std::vector<heap_element> heap;

    //! The map used to map from items to indexes in the heap.
    index_map_type index_map;

    //! Returns the index of the left child of the supplied index.
    std::size_t left(std::size_t i) const {
      return 2 * i;
    }

    //! Returns the index of the right child of the supplied index.
    std::size_t right(std::size_t i) const {
      return 2 * i + 1;
    }

    //! Returns the index of the parent of the supplied index.
    std::size_t parent(std::size_t i) const {
      return i / 2;
    }

    //! Extracts the priority at a heap location.
    Priority priority_at(std::size_t i) {
      return heap[i].second;
    }

    //! Compares the priorities at two heap locations.
    bool less(std::size_t i, std::size_t j) {
      return heap[i].second < heap[j].second;
    }

    //! Swaps the heap locations of two elements.
    void swap(std::size_t i, std::size_t j) {
      std::swap(heap[i], heap[j]);
      index_map[heap[i].first] = i;
      index_map[heap[j].first] = j;
    }

    //! The traditional heapify function.
    void heapify(std::size_t i) {
      std::size_t l = left(i);
      std::size_t r = right(i);
      std::size_t s = size();
      std::size_t largest = i;
      if ((l <= s) && less(i, l))
        largest = l;
      if ((r <= s) && less(largest, r))
        largest = r;
      if (largest != i) {
        swap(i, largest);
        heapify(largest);
      }
    }

  public:
    //! Default constructor.
    mutable_queue()
      : heap(1, std::make_pair(T(), Priority())) { }

    //! Returns the number of elements in the heap.
    std::size_t size() const {
      return heap.size() - 1;
    }

    //! Returns true iff the queue is empty.
    bool empty() const {
      return size() == 0;
    }

    //! Returns true if the queue contains the given value
    bool contains(const T& item) const {
      return index_map.count(item) > 0;
    }

    //! Enqueues a new item in the queue.
    void push(const T& item, Priority priority) {
      heap.push_back(std::make_pair(item, priority));
      std::size_t i = size();
      index_map[item] = i;
      while ((i > 1) && (priority_at(parent(i)) < priority)) {
        swap(i, parent(i));
        i = parent(i);
      }
    }

    //! Accesses the item with maximum priority in the queue.
    const std::pair<T, Priority>& top() const {
      return heap[1];
    }

    /**
     * Removes the item with maximum priority from the queue, and
     * returns it with its priority.
     */
    std::pair<T, Priority> pop() {
      heap_element top = heap[1];
      if (size() != 1)
        swap(1, size());
      heap.pop_back();
      heapify(1);
      index_map.erase(top.first);
      return top;
    }

    //! Returns the weight associated with a key
    Priority get(const T& item) const {
      return heap[index_map.at(item)].second;
    }

    //! Returns the priority associated with a key
    Priority operator[](const T& item) const {
      return get(item);
    }

    /**
     * Updates the priority associated with a item in the queue. This
     * function fails if the item is not already present.
    */
    void update(const T& item, Priority priority) {
      // Verify that the item is currently in the queue
      typename index_map_type::const_iterator iter = index_map.find(item);
      assert(iter != index_map.end());
      // If it is already present update the priority
      std::size_t i = iter->second;
      heap[i].second = priority;
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

    //! Returns the values (key-priority pairs) in the priority queue
    const std::vector<heap_element>& values() const {
      return heap;
    }

    //! Clears all the values (equivalent to stl clear)
    void clear() {
      heap.clear();
      heap.push_back(std::make_pair(T(), Priority()));
      index_map.clear();
    }

    //! Remove an item from the queue.
    //! Note: The item MUST be in the queue.
    void remove(const T& item) {
      // Ensure that the element is in the queue
      typename index_map_type::iterator iter = index_map.find(item);
      assert(iter != index_map.end());
      std::size_t i = iter->second;
      swap(i, size());
      heap.pop_back();
      heapify(i);
      // erase the element from the index map
      index_map.erase(iter);
    }

    //! Remove an item from the queue if it is present.
    void remove_if_present(const T& item) {
      typename index_map_type::iterator iter = index_map.find(item);
      if (iter == index_map.end())
        return;
      std::size_t i = iter->second;
      swap(i, size());
      heap.pop_back();
      heapify(i);
      // erase the element from the index map
      index_map.erase(iter);
    }

    //! Increment the priority of an item in the queue by inc if it is present.
    //! @return  True iff the item was present.
    bool increment_if_present(const T& item, Priority inc) {
      typename index_map_type::iterator iter = index_map.find(item);
      if (iter == index_map.end())
        return false;
      std::size_t i = iter->second;
      Priority new_priority = heap[i].second + inc;
      heap[i].second = new_priority;
      while ((i > 1) && (priority_at(parent(i)) < new_priority)) {
        swap(i, parent(i));
        i = parent(i);
      }
      heapify(i);
      return true;
    }

  }; // class mutable_queue



  template <typename Priority>
  class mutable_queue<std::size_t, Priority> {
  public:

    //! An element of the heap.
    typedef typename std::pair<std::size_t, Priority> heap_element;

  protected:

    //! The storage type of the index map
    typedef std::vector<int> index_map_type;

    //! The heap used to store the elements. The first element is unused.
    std::vector<heap_element> heap;

    //! The map used to map from items to indexes in the heap.
    index_map_type index_map;

    //! Returns the index of the left child of the supplied index.
    std::size_t left(std::size_t i) const {
      return 2 * i;
    }

    //! Returns the index of the right child of the supplied index.
    std::size_t right(std::size_t i) const {
      return 2 * i + 1;
    }

    //! Returns the index of the parent of the supplied index.
    std::size_t parent(std::size_t i) const {
      return i / 2;
    }

    //! Extracts the priority at a heap location.
    Priority priority_at(std::size_t i) {
      return heap[i].second;
    }

    //! Compares the priorities at two heap locations.
    bool less(std::size_t i, std::size_t j) {
      return heap[i].second < heap[j].second;
    }

    //! Swaps the heap locations of two elements.
    void swap(std::size_t i, std::size_t j) {
      std::swap(heap[i], heap[j]);
      index_map[heap[i].first] = i;
      index_map[heap[j].first] = j;
    }

    //! The traditional heapify function.
    void heapify(std::size_t i) {
      std::size_t l = left(i);
      std::size_t r = right(i);
      std::size_t s = size();
      std::size_t largest = i;
      if ((l <= s) && less(i, l))
        largest = l;
      if ((r <= s) && less(largest, r))
        largest = r;
      if (largest != i) {
        swap(i, largest);
        heapify(largest);
      }
    }

  public:
    //! Default constructor.
    mutable_queue()
      : heap(1, std::make_pair(-1, Priority())) { }

    //! Returns the number of elements in the heap.
    std::size_t size() const {
      return heap.size() - 1;
    }

    //! Returns true iff the queue is empty.
    bool empty() const {
      return size() == 0;
    }

    //! Returns true if the queue contains the given value
    bool contains(const std::size_t& item) const {
      if (index_map.size() > item) {
        return index_map[item] >= 0;
      }
      else {
        return false;
      }
    }

    //! Enqueues a new item in the queue.
    void push(std::size_t item, Priority priority) {
      heap.push_back(std::make_pair(item, priority));
      std::size_t i = size();
      if (index_map.size() < item+1) {
        index_map.resize(item+1, -1);
      }
      index_map[item] = i;
      while ((i > 1) && (priority_at(parent(i)) < priority)) {
        swap(i, parent(i));
        i = parent(i);
      }
    }

    //! Accesses the item with maximum priority in the queue.
    const std::pair<std::size_t, Priority>& top() const {
      return heap[1];
    }

    /**
     * Removes the item with maximum priority from the queue, and
     * returns it with its priority.
     */
    std::pair<std::size_t, Priority> pop() {
      heap_element top = heap[1];
      if (size() != 1)
        swap(1, size());
      heap.pop_back();
      heapify(1);
      index_map[top.first] = -1;
      return top;
    }

    //! Returns the weight associated with a key
    Priority get(std::size_t item) const {
      return heap[index_map[item]].second;
    }

    //! Returns the priority associated with a key
    Priority operator[](std::size_t item) const {
      return get(item);
    }

    /**
     * Updates the priority associated with a item in the queue. This
     * function fails if the item is not already present.
    */
    void update(std::size_t item, Priority priority) {
      std::size_t i = index_map[item];
      heap[i].second = priority;
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
    void updating_insert_max(std::size_t item, Priority priority) {
      if(!contains(item))
        push(item, priority);
      else{
        double effective_priority = std::max(get(item), priority);
        update(item, effective_priority);
      }
    }

    //! Returns the values (key-priority pairs) in the priority queue
    const std::vector<heap_element>& values() const {
      return heap;
    }

    //! Clears all the values (equivalent to stl clear)
    void clear() {
      heap.clear();
      heap.push_back(std::make_pair(-1, Priority()));
      index_map.clear();
    }

    //! Remove an item from the queue
    void remove(std::size_t item) {
      std::size_t i = index_map[item];
      swap(i, size());
      heap.pop_back();
      heapify(i);
      // erase the element from the index map
      index_map[i] = -1;
    }
  }; // class mutable_queue

} // namespace libgm

#endif
