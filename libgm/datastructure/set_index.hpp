#ifndef LIBGM_SET_INDEX_HPP
#define LIBGM_SET_INDEX_HPP

#include <libgm/iterator/counting_output_iterator.hpp>
#include <libgm/iterator/map_key_iterator.hpp>
#include <libgm/range/iterator_range.hpp>

#include <algorithm>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace libgm {

  /**
   * An index over sets that efficiently processes intersection and
   * superset queries and which supports efficient insertion and
   * deletion of sets. Each set is associated with a handle
   * that is returned during the superset and intersection queries.
   *
   * \tparam Handle
   *         Handle associated with each set in the index.
   *         Must be DefaultConstructible, CopyConstructible, and Assignable.
   * \tparam Range
   *         A container accepted by the lookup methods.
   * \tparam Hash
   *         A hash function for the values stored in the Range
   *
   * \ingroup datastructure
   */
  template <typename Handle,
            typename Range,
            typename Hash = std::hash<typename Range::value_type> >
  class set_index {

    //! The vector type that stores elements in a sorted order.
    typedef std::vector<typename Range::value_type> vector_type;

    //! Maps each element to the vectors containing the element.
    typedef std::unordered_map<
      typename Range::value_type, std::unordered_map<Handle, vector_type*>, Hash
    > adjacency_map;

    // Constructors and destructors
    //==========================================================================
  public:
    //! The type of values stored in the container.
    typedef typename Range::value_type value_type;

    //! An iterator over the values stored in the index.
    typedef map_key_iterator<adjacency_map> value_iterator;

    // Constructors and destructors
    //==========================================================================
  public:
    //! Default constructor. Creates an empty set index.
    set_index() { }

    //! Copy constructor.
    set_index(const set_index& other) {
      *this = other;
    }

    //! Move constructor.
    set_index(set_index&& other) {
      swap(*this, other);
    }

    //! Assignment operator.
    set_index& operator=(const set_index& other) {
      clear();
      for (const auto& vec : other.sets_) {
        auto ptr = new std::vector<value_type>(*vec.second);
        sets_.emplace(vec.first, std::unique_ptr<vector_type>(ptr));
        for (value_type value : *vec.second) {
          adjacency_[value].emplace(vec.first, ptr);
        }
      }
      return *this;
    }

    //! Move assignment operator.
    set_index& operator=(set_index&& other) {
      swap(*this, other);
      return *this;
    }

    //! Swaps the content of two index sets in constant time.
    friend void swap(set_index& a, set_index& b) {
      swap(a.sets_, b.sets_);
      swap(a.adjacency_, b.adjacency_);
    }

    // Queries
    //==========================================================================

    //! Returns true if the index contains no sets.
    bool empty() const {
      return sets_.empty();
    }

    //! Returns the handle of the first stored element.
    Handle front() const {
      assert(!empty());
      return sets_.begin()->first;
    }

    //! Returns the values stored in this index.
    iterator_range<value_iterator> values() const {
      return { value_iterator(adjacency_.begin()),
               value_iterator(adjacency_.end()) };
    }

    //! Returns the number of values stored in this index.
    std::size_t num_values() const {
      return adjacency_.size();
    }

    /**
     * Returns a handle for any set that contains the specified value.
     * \throw std::out_of_range if no there is no set containing the value
     */
    Handle operator[](value_type value) const {
      return adjacency_.at(value).begin()->first;
    }

    /**
     * Returns the number of sets that contain a value.
     */
    std::size_t count(value_type value) const {
      auto it = adjacency_.find(value);
      return it != adjacency_.end() ? it->second.size() : 0;
    }

    /**
     * Returns the size of the intersection of two sets with given handles.
     */
    std::size_t intersection_size(Handle a, Handle b) {
      counting_output_iterator out;
      const vector_type& veca = *sets_.at(a);
      const vector_type& vecb = *sets_.at(b);
      return std::set_intersection(veca.begin(), veca.end(),
                                   vecb.begin(), vecb.end(),
                                   out).count();
    }

    /**
     * Returns the sorted intersection of two sets with given handles.
     */
    Range intersection(Handle a, Handle b) {
      Range result;
      const vector_type& veca = *sets_.at(a);
      const vector_type& vecb = *sets_.at(b);
      std::set_intersection(veca.begin(), veca.end(),
                            vecb.begin(), vecb.end(),
                            std::inserter(result, result.end()));
      return result;
    }

    /**
     * Intersection query.  The handle for each set in this index that
     * intersects the supplied range is written to the output iterator.
     */
    void intersecting_sets(const Range& range,
                           std::function<void(Handle)> visitor) const {
      // reserve enough elements in the result to avoid reallocation
      std::unordered_set<Handle> visited;
      std::size_t nelems = 0;
      for (value_type value : range) {
        nelems += adjacency(value).size();
      }
      visited.reserve(nelems);

      // compute the union of all neighbors
      for (value_type value : range) {
        for (const auto& p : adjacency(value)) {
          visited.insert(p.first);
        }
      }

      // now visit them in arbitrary order
      for (Handle handle : visited) {
        visitor(handle);
      }
    }

    /**
     * Maximal intersection query: find a set whose intersection with
     * the supplied range is maximal and non-zero.
     *
     * \return the handle for a set with a non-zero maximal intersection
     *         with the supplied set. Of the sets with the same
     *         intersection size, returns the smallest one.
     *         Otherwise, returns front().
     */
    Handle find_max_intersection(const Range& range) const {
      if (range.begin() == range.end()) { return front(); }

      // reserve enough elements in the result to avoid reallocation
      std::unordered_set<Handle> visited;
      std::size_t nelems = 0;
      for (value_type value : range) {
        nelems += adjacency(value).size();
      }
      visited.reserve(nelems);

      // determine the maximum intersection
      std::vector<value_type> vec = sorted(range);
      Handle result = Handle();
      std::size_t max_inter = 0;
      std::size_t min_size = std::numeric_limits<std::size_t>::max();
      for (value_type value : range) {
        for (const auto& p : adjacency(value)) {
          if (!visited.count(p.first)) {
            visited.insert(p.first);
            counting_output_iterator out;
            std::size_t intersection =
              std::set_intersection(p.second->begin(), p.second->end(),
                                    vec.begin(), vec.end(), out).count();
            if ((intersection > max_inter) ||
                (intersection == max_inter && p.second->size() < min_size)) {
              max_inter = intersection;
              min_size = p.second->size();
              result = p.first;
            }
          }
        }
      }

      return result;
    }

    /**
     * Superset query. The handle for each set in this index that
     * contains all the elements of the supplied range is written
     * to the output iterator.
     */
    void supersets(const Range& range,
                   std::function<void(Handle)> visitor) const {
      if (range.begin() == range.end()) {
        // every set in the index is a superset of an empty range
        for (const auto& p : sets_) {
          visitor(p.first);
        }
      } else {
        // pick one value and iterate over all sets that contain that value
        std::vector<value_type> vec = sorted(range);
        for (const auto& p : adjacency(vec.front())) {
          if (std::includes(p.second->begin(), p.second->end(),
                            vec.begin(), vec.end())) {
            visitor(p.first);
          }
        }
      }
    }

    /**
     * Minimal superset query: find the hanlde of a minimal set which is
     * a superset of the supplied range. If the supplied range is empty,
     * this operation may return any set.
     *
     * \return the handle for a minimal superset if a superset exists.
     *         Handle() if no superset exists.
     */
    Handle find_min_cover(const Range& range) const {
      if (empty()) {
        return Handle();
      } else if (range.begin() == range.end()) {
        return front();
      } else {
        std::vector<value_type> vec = sorted(range);
        std::size_t min_size = std::numeric_limits<std::size_t>::max();
        Handle result = Handle();
        for (const auto& p : adjacency(vec.front())) {
          if (p.second->size() < min_size &&
              std::includes(p.second->begin(), p.second->end(),
                            vec.begin(), vec.end())) {
            min_size = p.second->size();
            result = p.first;
          }
        }
        return result;
      }
    }


    /**
     * Returns true if there are no (non-strict) supersets of the supplied
     * set in this index.
     */
    bool is_maximal(const Range& range) const {
      if (range.begin() == range.end()) {
        return sets_.empty();
      } else {
        // pick one value and iterate over all sets that contain that value
        std::vector<value_type> vec = sorted(range);
        for (const auto& p : adjacency(vec.front())) {
          if (std::includes(p.second->begin(), p.second->end(),
                            vec.begin(), vec.end())) {
            return false;
          }
        }
        return true;
      }
    }

    /**
     * Prints the index to an output stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const set_index& index) {
      for (const auto& sets : index.adjacency_) {
        out << sets.first << " -->";
        for (const auto& p : sets.second) {
          out << ' ' << p.first;
        }
        out << std::endl;
      }
      return out;
    }

    // Mutating operations
    //==========================================================================

    /**
     * Inserts a new set in the index. This function is linear in the
     * number of elements of the range. The range must not be empty, and
     * the handle must not be yet present in the index.
     */
    void insert(Handle handle, const Range& range) {
      if (sets_.count(handle)) {
        throw std::invalid_argument("Handle already present");
      }
      auto ptr = new std::vector<value_type>(sorted(range));
      sets_.emplace(handle, std::unique_ptr<vector_type>(ptr));
      for (value_type value : range) {
        adjacency_[value].emplace(handle, ptr);
      }
    }

    //! Removes the set with the given handle from the index.
    void erase(Handle handle) {
      auto it = sets_.find(handle);
      assert(it != sets_.end());
      for (value_type value : *it->second) {
        adjacency_[value].erase(handle);
        if (adjacency_[value].empty()) {
          adjacency_.erase(value);
        }
      }
      sets_.erase(it); // automatically frees memory
    }

    //! Removes all sets from this index.
    void clear() {
      sets_.clear();
      adjacency_.clear();
    }

    // Private members
    //==========================================================================
  private:

    //! Converts a range to a vector with sorted elements.
    vector_type sorted(const Range& range) const {
      vector_type vec(range.begin(), range.end());
      std::sort(vec.begin(), vec.end());
      vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
      return vec;
    }

    //! Returns the edges from a single value to the sets it is contained in/
    const std::unordered_map<Handle, vector_type*>&
    adjacency(value_type value) const {
      static std::unordered_map<Handle, vector_type*> empty;
      auto it = adjacency_.find(value);
      if (it == adjacency_.end()) {
        return empty;
      } else {
        return it->second;
      }
    }

    //! Stores the mapping from handles to sets.
    std::unordered_map<Handle, std::unique_ptr<vector_type> > sets_;

    //! Stores the mapping from values to sets.
    adjacency_map adjacency_;

  }; // class set_index

} // namespace libgm

#endif
