#pragma once

#include <libgm/argument/shape.hpp>
#include <libgm/datastructure/subrange.hpp>
#include <libgm/iterator/table_index_iterator.hpp>

#include <numeric>

namespace libgm {

/**
 * A dense table with an arbitrary number of dimensions, each with a
 * finite number of values. The elements are stored in a linear fashion,
 * with the lowest-dimension indices changing the most frequently and
 * the highest-dimension indices changing the least. Thus, a table with
 * dimension (3,2) stores the elements in the order (0,0), (1,0), (2,0),
 * (0,1), (1,1), (2,1).
 *
 * \ingroup datastructure
 */
template <typename T>
class Table {
public:
  // Public type declarations
  //--------------------------------------------------------------------------

  /// The type of values stored in this object
  typedef T value_type;

  /// An iterator over the elements of this table in a linear fashion
  typedef T* iterator;

  /// A const iterator over the elements of this table in a linear fashion
  typedef const T* const_iterator;

  // Constructors and initialization
  //--------------------------------------------------------------------------

  /**
   * Default constructor. Creates an empty table with no elements.
   */
  Table() = default;

  /**
   * Constructs a table with the given shape. This constructor does not
   * initialize the table elements. If needed, the table contents can be
   * initialized with other constructors or with the fill() function.
   */
  explicit Table(Shape shape) {
    reset(std::move(shape));
  }

  /**
   * Constructs a table with the given shape and initializes its elements
   * to the given value.
   */
  Table(Shape shape, const T& init) {
    reset(std::move(shape));
    fill(init);
  }

  /**
   * Constructs a table with the given shape and initializes its elements
   * to the given list.
   */
  Table(Shape shape, std::initializer_list<T> values) {
    reset(std::move(shape));
    assert(size() == values.size());
    std::copy(values.begin(), values.end(), data());
  }

  /**
   * Constructs a table with the given shape and contents of the range.
   */
  template <typename It>
  Table(Shape shape, It begin, It end) {
    reset(std::move(shape));
    assert(size() == std::distance(begin, end));
    std::copy(begin, end, data());
  }

  /**
   * Copy constructor. Copies the shape and elements of a table to this one.
   */
  Table(const Table& x)
    : shape_(x.shape_), size_(x.size_) {
    data_.reset(new T[size()]);
    std::copy(x.begin(), x.end(), data());
  }

  /// Move constructor
  Table(Table&& x) {
    swap(*this, x);
  }

  /// Assignment operator. Copies the shape and elements from x to this table.
  Table& operator=(const Table& x) {
    if (this != &x) {
      if (size() != x.size()) { data_.reset(new T[x.size()]); }
      shape_ = x.shape_;
      size_ = x.size_;
      std::copy(x.begin(), x.end(), data());
    }
    return *this;
  }

  Table& operator=(Table&& x) {
    swap(*this, x);
    return *this;
  }

  /// Swaps the contents of two tables.
  friend void swap(Table& x, Table& y) {
    if (&x != &y) {
      using std::swap;
      swap(x.shape_, y.shape_);
      swap(x.size_, y.size_);
      swap(x.data_, y.data_);
    }
  }

  /**
   * Reset the table to the given shape. Allocates the memory, but does
   * not initialize the elements.
   */
  void reset(Shape shape) {
    size_t old_size = size_;
    size_ = shape.product();
    shape_ = std::move(shape);
    if (size_ != old_size) {
      data_.reset(new T[size_]);
    }
  }

  /**
   * Sets all the elements to the given value.
   */
  void fill(const T& value) {
    std::fill(begin(), end(), value);
  }

  /**
   * Verifies that the specified shape matches this table.
   */
  void check_shape(const Shape& shape) const {
    if (shape_ != shape) {
      throw std::invalid_argument("Incompatible shapes");
    }
  }

  // Accessors
  //--------------------------------------------------------------------------

  /// Returns the number of dimensions of this table.
  size_t arity() const {
    return shape_.size();
  }

  /// Returns the dimensions of this table.
  const Shape& shape() const {
    return shape_;
  }

  /// Returns the size of the given dimension of this table
  size_t size(size_t n) const {
    return shape_[n];
  }

  /// Total number of elements in the table.
  size_t size() const {
    return size_;
  }

  /// Returns true if the table was default-initialized, i.e. has no elements.
  bool empty() const {
    return !data_;
  }

  /// Returns a pointer to the allocated values or null if the table is empty.
  T* data() {
    return data_.get();
  }

  /// Returns a pointer to the allocated values or null if the table is empty.
  const T* data() const {
    return data_.get();
  }

  /// Returns the pointer (iterator) to the first element.
  T* begin() {
    return data_.get();
  }

  /// Returns the pointer (iterator) to the first element.
  const T* begin() const {
    return data_.get();
  }

  /// Returns the pointer (iterator) to the one past the last element.
  T* end() {
    return data_.get() + size_;
  }

  /// Returns the pointer (iterator) to the one past the last element.
  const T* end() const {
    return data_.get() + size_;
  }

  /// Returns the index associated with an iterator position
  std::vector<size_t> index(const T* it) const {
    std::vector<size_t> result;
    shape_.index(it - begin(), result);
    return result;
  }

  /// Returns an iterator range over indices into this table.
  SubRange<TableIndexIterator> indices() const {
    if (empty()) {
      return { TableIndexIterator(), TableIndexIterator() };
    } else {
      return { TableIndexIterator(shape_), TableIndexIterator(arity()) };
    }
  }

  /// Returns the element with the specified linear index
  T& operator[](size_t i) {
    return data_[i];
  }

  /// Returns the element with the specified linear index
  const T& operator[](size_t i) const {
    return data_[i];
  }

  /// Returns the element with the specified finite index
  T& operator()(const std::vector<size_t>& index) {
    return data_[shape_.linear(index)];
  }

  /// Returns the element with the specified finite index
  const T& operator()(const std::vector<size_t>& index) const {
    return data_[shape_.linear(index)];
  }

  /// Returns true if the two tables have the same shape and elements
  bool operator==(const Table& x) const {
    return shape_ == x.shape_ && std::equal(begin(), end(), x.begin());
  }

  /// Returns true if the two tables do not have the same shape or elements
  bool operator!=(const Table& x) const {
    return !(*this == x);
  }

  // Private members
  //--------------------------------------------------------------------------

  /// The dimensions of this table.
  Shape shape_;

  /// The total number of elements.
  size_t size_ = 0;

  /// The elements of this table, stored in a linear order.
  std::unique_ptr<T[]> data_;

}; // class Table

template <typename T>
std::ostream& operator<<(std::ostream& out, const Table<T>& table) {
  std::ostream_iterator<std::size_t, char> out_it(out, " ");
  const T* elem = table.data();
  for (const std::vector<size_t>& index : table.indices()) {
    std::copy(index.begin(), index.end(), out_it);
    out << *elem++ << std::endl;
  }
  return out;
}

} // namespace libgm
