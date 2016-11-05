#ifndef LIBGM_INDEX_RANGE_HPP
#define LIBGM_INDEX_RANGE_HPP

#include <libgm/enable_if.hpp>
#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/iterator/counting_iterator.hpp>
#include <libgm/parser/range_io.hpp>

#include <algorithm>
#include <iosfwd>
#include <numeric>
#include <type_traits>

namespace libgm {

  /**
   * A range of indices specified as a starting random access iterator
   * and size.
   */
  template <typename It>
  class index_range {
  public:
    //! Indicates whether this range is guaranteed to be contiguous.
    static const bool is_contiguous =
      std::is_same<It, counting_iterator>::value;

    //! Creates an empty range.
    index_range()
      : begin_(), size_(0) { }

    //! Creates a range given a starting iterator and size.
    index_range(It begin, std::size_t size)
      : begin_(begin), size_(size) { }

    //! Creates a reference to an uint_vector.
    LIBGM_ENABLE_IF((std::is_same<It, const std::size_t*>::value))
    explicit index_range(const uint_vector& indices)
      : begin_(indices.data()), size_(indices.size()) { }

    //! Returns the size of the range.
    std::size_t size() const {
      return size_;
    }

    //! Returns true if the range is empty.
    bool empty() const {
      return size_ == 0;
    }

    //! Returns the specified index in the range.
    std::size_t operator[](std::size_t i) const {
      return begin_[i];
    }

    //! Returns the iterator to the first index in the range.
    It begin() const {
      return begin_;
    }

    //! Returns the iterator past the last index in the range.
    It end() const {
      return begin_ + size_;
    }

    //! Returns the first index of a non-empty range.
    std::size_t front() const {
      assert(!empty());
      return *begin();
    }

    //! Returns the minimum index of the range or 0 if empty.
    std::size_t start() const {
      return is_contiguous
        ? *begin()
        : (empty() ? 0 : *std::min_element(begin(), end()));
    }

    //! Returns the maximum index of the range + 1 or 0 if empty.
    std::size_t stop() const {
      return is_contiguous
        ? *end()
        : (empty() ? 0 : *std::max_element(begin(), end()) + 1);
    }

    //! Returns true if all the index in the range are less than a given index.
    friend bool operator<(index_range r, std::size_t bound) {
      return r.empty() || *std::max_element(r.begin(), r.end()) < bound;
    }

    //! Prints the range to an output stream.
    friend std::ostream& operator<<(std::ostream& out, index_range range) {
      print_range(out, range.begin(), range.end(), '[', ',', ']');
      return out;
    }

    //! Swaps two index ranges.
    friend void swap(index_range& x, index_range& y) {
      using std::swap;
      swap(x.begin_, y.begin_);
      swap(x.size_, y.size_);
    }

  protected:
    //! The iterator to the first index.
    It begin_;

    //! The size of the range.
    std::size_t size_;

  }; // class index_range

  /**
   * A contiguous range of indices.
   * \relates index_range
   */
  using span = index_range<counting_iterator>;

  /**
   * A list of indices stored in a block of memory.
   * \relates index_range
   */
  using iref = index_range<const std::size_t*>;


  // Specialized ranges
  //============================================================================

  /**
   * A special type of span that designates leading indices.
   */
  class front : public span {
  public:
    //! Creates a front range with the given number of indices.
    explicit front(std::size_t size)
      : span(0, size) { }

    friend std::ostream& operator<<(std::ostream& out, front f) {
      out << "front(" << f.size() << ")";
      return out;
    }
  }; // class front


  /**
   * A special type of span that designates trailing indices.
   */
  class back : public span {
  public:
    //! Creates a back range with the end index and size.
    back(std::size_t stop, std::size_t size)
      : span(stop - size, size) {
      assert(size <= stop);
    }

    friend std::ostream& operator<<(std::ostream& out, back b) {
      out << "back(" << b.stop() << "," << b.size() << ")";
      return out;
    }
  }; // class back


  /**
   * A special type of span that designates all dimensions.
   */
  class all : public span {
  public:
    //! Creates an all-range with the given size.
    all(std::size_t size)
      : span(0, size) { }

    friend std::ostream& operator<<(std::ostream& out, all a) {
      out << "all(" << a.size() << ")";
      return out;
    }

  }; // class all


  /**
   * A special type of span that designates a single value.
   */
  class single : public span {
  public:
    //! Creates a range with the given single value.
    explicit single(std::size_t value)
      : span(value, 1) { }

    //! Returns the unique value represented by this range.
    std::size_t value() const {
      return start();
    }

    friend std::ostream& operator<<(std::ostream& out, single s) {
      out << "single(" << s.value() << ")";
      return out;
    }
  }; // class single


  /**
   * A special type of iref that manages its elements.
   */
  class ivec : public iref {
  public:
    //! Constructs an uninitialized vector with the given size.
    explicit ivec(std::size_t size)
      : iref(size > 0 ? new std::size_t[size] : nullptr, size) { }

    //! Consructs a vector with the given size, initialized to a value.
    ivec(std::size_t size, std::size_t init)
      : ivec(size) {
      fill(init);
    }

    //! Constructs a vector with the given elements.
    ivec(std::initializer_list<std::size_t> init)
      : ivec(init.size()) {
      std::copy(init.begin(), init.end(), data());
    }

    //! Move constructor.
    ivec(ivec&& other) {
      swap(*this, other);
    }

    //! Frees the memory associated with this vector.
    ~ivec() {
      if (data()) {
        delete[] data();
      }
    }

    //! Disallow copy.
    ivec(const ivec& other) = delete;

    //! Returns the pointer to the data.
    std::size_t* data() {
      return const_cast<std::size_t*>(begin_);
    }

    //! Mutable access to the index.
    std::size_t& operator[](std::size_t i) {
      return data()[i];
    }

    //! Returns an iterator to the first index.
    std::size_t* begin() {
      return data();
    }

    //! Returns an iterator past the last index.
    std::size_t* end() {
      return data() + size();
    }

    //! Fills the vector with the given value.
    void fill(std::size_t value) {
      std::fill_n(data(), size(), value);
    }

    //! Reduces the size of the vector.
    void trim(std::size_t new_size) {
      assert(new_size <= size());
      size_ = new_size;
    }

  }; // class ivec


  // Additional functions for span
  //============================================================================

  /**
   * Returns a span with the given boundaries.
   */
  inline span bounds(std::size_t start, std::size_t stop) {
    assert(start <= stop);
    return { start, stop - start };
  }

  /**
   * Prints the span to an output stream.
   * \relates span
   */
  inline std::ostream& operator<<(std::ostream& out, span s) {
    out << '[' << s.start() << ';' << s.stop() << ')';
    return out;
  }

  /**
   * Returns true if two spans have a contiguous union.
   * \relates span
   */
  inline bool contiguous_union(span s, span t) {
    return t.start() <= s.stop() && t.stop() >= s.start();
  }

  /**
   * Returns the union of two spans, provided that the union is contiguous.
   * \relates span
   */
  inline span operator|(span s, span t) {
    assert(contiguous_union(s, t));
    return bounds(std::min(s.start(), t.start()),
                  std::max(s.stop(), t.stop()));
  }

  /**
   * Returns the concatenation of two spans.
   * \relats span
   */
  inline uint_vector operator+(span s, span t) {
    uint_vector result(s.size() + t.size());
    std::iota(result.begin(), result.begin() + s.size(), s.start());
    std::iota(result.begin() + s.size(), result.end(), t.start());
    return result;
  }

} // namespace libgm

#endif
