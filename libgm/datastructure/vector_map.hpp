#ifndef LIBGM_VECTOR_MAP_HPP
#define LIBGM_VECTOR_MAP_HPP

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace libgm {

  /**
   * A map that stores the key-value pairs in an std::vector.
   * The pairs need to be explicitly sorted using the sort()
   * member before the find() function can be invoked.
   *
   * \ingroup datastructure
   */
  template <typename Key,
            typename T,
            typename Compare = std::less<Key> >
  class vector_map {
    typedef std::vector<std::pair<Key,T>> vec_type;
  public:
    // Associative container types
    typedef Key               key_type;
    typedef T                 mapped_type;
    typedef std::pair<Key, T> value_type;
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;
    typedef Compare           key_compare;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;

    typedef typename vec_type::iterator       iterator;
    typedef typename vec_type::const_iterator const_iterator;

    struct value_compare {
    private:
      Compare comp;
    public:
      value_compare(Compare comp) : comp(comp) { }
      bool operator()(const value_type& a,
                      const value_type& b) const {
        return comp(a.first, b.first);
      }
      bool operator()(const value_type& a, const Key& b) const {
        return comp(a.first, b);
      }
      bool operator()(const Key& a, const value_type& b) const {
        return comp(a, b.first);
      }
    };

    // Constructors and assignment
    //==========================================================================

    //! Default constructor. Constructs an empty map.
    explicit vector_map(const Compare& comp = Compare())
      : comp_(comp), sorted_(true) { }

    //! Constructs the map with the contents of the given range.
    template <typename It>
    vector_map(It first, It last, const Compare& comp = Compare())
      : comp_(comp), sorted_(true) {
      insert(first, last);
      sort();
    }

    //! Constructs the map with the contents of the initializer list.
    vector_map(std::initializer_list<value_type> init,
               const Compare& comp = Compare())
      : comp_(comp), sorted_(true) {
      insert(init);
      sort();
    }

    //! Replaces the contents of this container with the given list.
    vector_map& operator=(std::initializer_list<value_type> init) {
      clear();
      insert(init);
      sort();
      return *this;
    }

    //! Swaps the contents of two containers.
    friend void swap(vector_map& a, vector_map& b) {
      using std::swap;
      swap(a.elems_, b.elems_);
      swap(a.comp_, b.comp_);
      swap(a.sorted_, b.sorted_);
    }

    // Accessors
    //==========================================================================

    //! Returns the number of elements in the map.
    std::size_t size() const {
      return elems_.size();
    }

    //! Returns true if the map is empty.
    bool empty() const {
      return elems_.empty();
    }

    //! Returns true if the map is sorted.
    bool sorted() const {
      return sorted_;
    }

    //! Returns an iterator to the first element of the map.
    iterator begin() {
      return elems_.begin();
    }

    //! Returns an iterator to the first element of the map.
    const_iterator begin() const {
      return elems_.begin();
    }

    //! Returns an iterator to the element past the last element of the map.
    iterator end() {
      return elems_.end();
    }

    //! Returns an iterator to the element past the last element of the map.
    const_iterator end() const {
      return elems_.end();
    }

    /**
     * Returns a reference to the mapped type for the given key.
     * \throws std::out_of_range if the map does not have an element
     *         with the specified key
     */
    T& at(const Key& key) {
      iterator it = find(key);
      if (it == end()) {
        throw std::out_of_range("vector_map::at: missing key");
      }
      return it->second;
    }

    /**
     * Returns a const-reference to the mapped type for the given key.
     * \throws std::out_of_range if the map does not have an element
     *         with the specified key
     */
    const T& at(const Key& key) const {
      const_iterator it = find(key);
      if (it == end()) {
        throw std::out_of_range("vector_map::at: missing key");
      }
      return it->second;
    }

    //! Returns the number of elements matching the given key.
    std::size_t count(const Key& key) const {
      return find(key) != end();
    }

    //! Finds an element with the given key.
    iterator find(const Key& key) {
      iterator it = lower_bound(key);
      return (it != end() && it->first == key) ? it : end();
    }

    //! Find an element with the given key.
    const_iterator find(const Key& key) const {
      const_iterator it = lower_bound(key);
      return (it != end() && it->first == key) ? it : end();
    }

    //! Returns a range containing all elements with the given key.
    std::pair<iterator, iterator>
    equal_range(const Key& key) {
      iterator it = lower_bound(key);
      if (it != end() && it->first == key) {
        return {it, std::next(it)};
      } else {
        return {it, it};
      }
    }

    //! Returns a range containing all elements with the given key.
    std::pair<const_iterator, const_iterator>
    equal_range(const Key& key) const {
      const_iterator it = lower_bound(key);
      if (it != end() && it->first == key) {
        return {it, std::next(it)};
      } else {
        return {it, it};
      }
    }

    //! Returns an iterator to the first element not less than key.
    iterator lower_bound(const Key& key) {
      assert(sorted_);
      return std::lower_bound(begin(), end(), key, value_comp());
    }

    //! Returns an iterator to the first element not less than key.
    const_iterator lower_bound(const Key& key) const {
      assert(sorted_);
      return std::lower_bound(begin(), end(), key, value_comp());
    }

    //! Returns an iterator to the first element greater than key.
    iterator upper_bound(const Key& key) {
      assert(sorted_);
      return std::upper_bound(begin(), end(), key, value_comp());
    }

    //! Returns an iterator to the first element greater than key.
    const_iterator upper_bound(const Key& key) const {
      assert(sorted_);
      return std::upper_bound(begin(), end(), key, value_comp());
    }

    //! Returns the function object that compares keys.
    key_compare key_comp() const {
      return comp_;
    }

    //! Returns the function object that compares key in values_type.
    value_compare value_comp() const {
      return value_compare(comp_);
    }

    //! Returns true if two sorted vector_map objects are equal.
    friend bool operator==(const vector_map& a, const vector_map& b) {
      assert(a.sorted() && b.sorted());
      return a.elems_ == b.elems_;
    }

    //! Returns true if two sorted vector_map objects are not equal.
    friend bool operator!=(const vector_map& a, const vector_map& b) {
      return !(a == b);
    }

    // Modification
    //==========================================================================

    //! Clears the contents of the map.
    void clear() {
      elems_.clear();
      sorted_ = true;
    }

    //! Ensures that the vector map has space for at least n elements.
    void reserve(std::size_t n) {
      elems_.reserve(n);
    }

    //! Inserts a value to the map (may clear the sorted flag).
    void insert(const value_type& value) {
      if (!empty() && !comp_(elems_.back().first, value.first)) {
        sorted_ = false;
      }
      elems_.push_back(value);
    }

    //! Inserts a value to the map, ignoring the hint.
    iterator insert(const_iterator /* hint */, const value_type& value) {
      insert(value);
      return begin();
    }

    //! Inserts elements from the given range (may clear the sorted flag).
    template <typename It>
    void insert(It first, It last) {
      std::copy(first, last, std::inserter(*this, begin()));
    }

    //! Inserts elements from an initializer list (may clear the sorted flag).
    void insert(std::initializer_list<value_type> list) {
      std::copy(list.begin(), list.end(), std::inserter(*this, begin()));
    }

    //! Insert a new element to the map constructed in-place.
    template <typename... Args>
    void emplace(Args&&... args) {
      bool prev_empty = empty();
      elems_.emplace_back(std::forward<Args>(args)...);
      auto it = --end();
      if (!prev_empty && !comp_((it-1)->first, it->first)) {
        sorted_ = false;
      }
    }

    //! Removes the element at the given position.
    iterator erase(const_iterator pos) {
      return elems_.erase(pos);
    }

    //! Removes the elements in the given range.
    iterator erase(const_iterator first, const_iterator last) {
      return elems_.erase(first, last);
    }

    //! Removes all elements with the key value key.
    std::size_t erase(const Key& key) {
      iterator first, last;
      std::tie(first, last) = equal_range(key);
      std::size_t n = std::distance(first, last);
      elems_.erase(first, last);
      return n;
    }

    //! Substitutes keys in place according to the given map.
    template <typename Map>
    void subst_keys(const Map& map) {
      for (auto& p : *this) {
        p.first = map.at(p.first);
      }
      sorted_ = false;
      sort();
    }

    //! Sorts the contents of the container if needed.
    void sort() {
      if (sorted_) return;
      std::sort(begin(), end(), value_comp());
      sorted_ = true;
    }

  private:
    std::vector<std::pair<Key,T>> elems_;
    Compare comp_;
    bool sorted_;

  }; // class vector_map

  /**
   * Prints the map to a stream.
   * \relates vector_map
   */
  template <typename Key, typename T>
  std::ostream&
  operator<<(std::ostream& out, const vector_map<Key, T>& map) {
    out << "{";
    bool first = true;
    for (const auto& p : map) {
      if (first) { first = false; } else { out << ", "; }
      out << p.first << ":" << p.second;
    }
    out << "}";
    return out;
  }

} // namespace libgm

#endif
