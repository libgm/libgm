#ifndef LIBGM_INDEX_MAP_HPP
#define LIBGM_INDEX_MAP_HPP

#include <libgm/global.hpp>

#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace libgm {

  /**
   * A class that represents a (possibly contiguous) sequence of indices
   * of a vector or a matrix.
   *
   * \see submatrix
   */
  class matrix_index {
  public:
    //! Constructs an empty index.
    matrix_index()
      : start_(0), size_(0) { }

    //! Constructs a contiguous index [start; start + size).
    matrix_index(size_t start, size_t size)
      : start_(start), size_(size) { }

    //! Constructs a non-contiguous index with given elements.
    matrix_index(std::initializer_list<size_t> init)
      : start_(0), size_(0), indices_(init) { }

    //! Returns true if the index is contiguous.
    bool contiguous() const {
      return indices_.empty();
    }

    //! Returns true if the index contains no elements.
    bool empty() const {
      return size_ == 0 && indices_.empty();
    }

    //! Returns the start if the index is contiguous and 0 otherwise.
    size_t start() const {
      return start_;
    }

    //! Returns the number of elements in the index.
    size_t size() const {
      return size_ + indices_.size();
    }

    //! Returns the i-th element in a non-contiguous index.
    size_t operator[](size_t i) const {
      assert(!indices_.empty());
      return indices_[i];
    }

    //! Returns the i-th element in the index.
    size_t operator()(size_t i) const {
      return indices_.empty() ? start_ + i : indices_[i];
    }
    
    /**
     * Adds a range to the index. Attempts to preserve the contiguity
     * of the index, i.e., if the index is presently contiguous with
     * [old_start; start), the index will be still contiguous after
     * a call to this function with [old_start; start + size).
     */
    void append(size_t start, size_t size) {
      if (size == 0) {        // ignore
        return;
      } else if (empty()) {   // previously empty
        start_ = start;
        size_ = size;
      } else if (size_ > 0) { // previouly contiguous
        if (start_ + size_ == start) {
          size_ += size;
        } else {
          indices_.resize(size_ + size);
          std::iota(indices_.begin(), indices_.begin() + size_, start_);
          std::iota(indices_.begin() + size_, indices_.end(), start);
          start_ = 0;
          size_ = 0;
        }
      } else {                // previously non-contiguous
        indices_.resize(indices_.size() + size);
        std::iota(indices_.end() - size, indices_.end(), start);
      }
    }

    //! Swaps the contents of this index with another one.
    void swap(matrix_index& other) {
      using std::swap;
      swap(start_, other.start_);
      swap(size_, other.size_);
      swap(indices_, other.indices_);
    }

  private:
    size_t start_, size_;
    std::vector<size_t> indices_;

  }; // class matrix_index

  /**
   * Outputs a matrix_index to an output stream.
   * \relates matrix_index
   */
  inline std::ostream&
  operator<<(std::ostream& out, const matrix_index& index) {
    if (index.contiguous()) {
      out << '[' << index.start()
          << "; " << index.start() + index.size()
          << ')';
    } else {
      out << '{';
      for (size_t i = 0; i < index.size(); ++i) {
        if (i > 0) out << ", ";
        out << index[i];
      }
      out << '}';
    }
    return out;
  }
  
  /**
   * Swaps the contents of two matrix_index objects.
   * \relates matrix_index
   */
  inline void swap(matrix_index& a, matrix_index& b) {
    a.swap(b);
  }

} // namespace libgm

#endif
