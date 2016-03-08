#ifndef LIBGM_TABLE_HPP
#define LIBGM_TABLE_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/datastructure/uint_vector_iterator.hpp>
#include <libgm/functional/algorithm.hpp>
#include <libgm/functional/arithmetic.hpp>
#include <libgm/functional/iterator.hpp>
#include <libgm/functional/nth_value.hpp>
#include <libgm/functional/tuple.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>
#include <libgm/traits/missing.hpp>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace libgm {

  /**
   * A class that can perform conversions between uint_vector and linear
   * index in a dense table. It stores the multipliers for each dimension
   * and the total size of the table.
   *
   * \ingroup datastructure
   */
  class table_offset {
  public:
    /**
     * Constructs a table_offset object for an empty shape.
     * Initializes the size to 0.
     */
    table_offset() : nelem_(0) { }

    /**
     * Constructs a table_offset object for the table with the given shape.
     * The number of dimensions may be zero (i.e., the shape may be an
     * empty vector), but none of the dimensions themselves may be zero
     * (i.e., the shape may not contain any zero elements).
     */
    explicit table_offset(const uint_vector& shape) {
      reset(shape);
    }

    /**
     * Swaps this object with another.
     */
    friend void swap(table_offset& x, table_offset& y) {
      std::swap(x.multiplier_, y.multiplier_);
      std::swap(x.nelem_, y.nelem_);
    }

    //! Resets the table_offset object to the given shape.
    void reset(const uint_vector& shape) {
      multiplier_.resize(shape.size());
      nelem_ = 1;
      for (std::size_t i = 0; i < shape.size(); ++i) {
        multiplier_[i] = nelem_;
        nelem_ *= shape[i];
      }
    }

    //! Returns the number of dimensions of this offset.
    std::size_t size() const {
      return multiplier_.size();
    }

    //! Returns the total number of elements occupied by the table.
    std::size_t num_elements() const {
      return nelem_;
    }

    //! Returns the multiplier vector.
    const std::vector<std::size_t>& multiplier() const {
      return multiplier_;
    }

    //! Returns the multiplier associated with the dimension d <= size().
    std::size_t multiplier(std::size_t d) const {
      return (d == size()) ? nelem_ : multiplier_[d];
    }

    //! Calculates the linear index for the given finite index.
    std::size_t linear(const uint_vector& index) const {
      assert(multiplier_.size() == index.size());
      std::size_t result = 0;
      for (std::size_t i = 0; i < multiplier_.size(); ++i) {
        result += multiplier_[i] * index[i];
      }
      return result;
    }

    /**
     * Calculates the linear index for the given partial finite index.
     * The finite index is assumed to start at pos-th dimension and may
     * be shorter than the remaining dimensions. The finite values in the
     * omitted dimensions are assumed to be 0.
     */
    std::size_t linear(const uint_vector& index, std::size_t pos) const {
      assert(index.size() + pos <= multiplier_.size());
      std::size_t result = 0;
      for (std::size_t i = 0; i < index.size(); ++i) {
        result += multiplier_[i + pos] * index[i];
      }
      return result;
    }

    std::size_t linear(const uint_vector& dims, const uint_vector& vals) const {
      assert(dims.size() == vals.size());
      std::size_t result = 0;
      for (std::size_t i = 0; i < dims.size(); ++i) {
        result += multiplier_[dims[i]] * vals[i];
      }
      return result;
    }

    /**
     * Computes the vector index for the given linear offset.
     * \throw std::out_of_range
     *        if the offset is larger than the number of elements in the table
     */
    void vector(std::size_t offset, uint_vector& index) const {
      vector(offset, 0, size(), index);
    }

    /**
     * Computes the vector index for the given linear offset for the first
     * n dimensions.
     * \throw std::out_of_range
     *        if the offset is larger than the number of elements in the first
     *        n dimensions
     */
    void vector(std::size_t offset, std::size_t n, uint_vector& index) const {
      vector(offset, 0, n, index);
    }

    /**
     * Computes the vector index for the given linear offset for dimensions
     * m, ..., n-1.
     * \throw std::out_of_range
     *        if the offset is larger than the number of elements in the first
     *        n dimensions
     */
    void vector(std::size_t offset, std::size_t m, std::size_t n,
                uint_vector& index) const {
      assert(n <= size());
      if (offset >= multiplier(n)) {
        throw std::out_of_range("Table offset is out of range");
      }
      index.resize(n-m);
      for (std::ptrdiff_t i = n - 1; i >= ptrdiff_t(m); --i) {
        index[i-m] = offset / multiplier_[i];
        offset %= multiplier_[i];
      }
    }

  private:
    //! The multiplier associated with the index in each dimension.
    std::vector<std::size_t> multiplier_;

    //! The total number of elements in the underlying table.
    std::size_t nelem_;

  }; // class table_offset


  /**
   * A utility class that represents the increments to the linear index
   * when iterating over multiple tables in parallel. The setting is as
   * follows. We have two tables: table A that is being iterated over in
   * a linear fashion, and table B from which we select the corresponding
   * elements in a "jumpy" fashion. The d-th element of this class
   * represents the additive change (positive, zero, or negative) to
   * table B's linear index, as we increase digit d of table A's
   * finite index by 1.
   */
  class table_increment {
  public:
    /**
     * Creates a table increment for the given shape of table A,
     * the offset of table B, and mapping of dimensions of table B
     * to table A.
     */
    table_increment(const uint_vector& a_shape,
                    const table_offset& b_offset,
                    const uint_vector& b_map)
      : inc_(a_shape.size() + 1, 0) { // do we need + 1?
      assert(b_offset.size() == b_map.size());
      for (std::size_t i = 0; i < b_map.size(); ++i) {
        if (b_map[i] != missing<std::size_t>::value) {
          inc_[b_map[i]] = b_offset.multiplier(i);
        }
      }
      for (std::size_t i = a_shape.size(); i > 0; --i) {
        inc_[i] -= inc_[i-1] * a_shape[i-1];
      }
    }

    /**
     * Creates a table increment corresponding to a linear scan through
     * a table.
     */
    table_increment(const uint_vector& a_shape, const table_offset& b_offset)
      : inc_(a_shape.size() + 1, 0) { // need + 1 in case a is constant
      inc_[0] = 1;
      inc_[b_offset.size()] = -ptrdiff_t(b_offset.num_elements());
    }

    /**
     * Updates the increment to contain a partial (cumulative) sum
     * of its present values.
     */
    void partial_sum(std::size_t start = 0) {
      std::partial_sum(inc_.begin() + start, inc_.end(), inc_.begin() + start);
    }

    /**
     * Returns the increment for the given dimension.
     */
    std::ptrdiff_t operator[](std::size_t i) const {
      return inc_[i];
    }

  private:
    std::vector<std::ptrdiff_t> inc_;

  }; // class table_increment


  /**
   * A function that loops linearly over one table and non-linearly
   * over another tables.
   *
   * This function implements a hybrid approach, where the two lowest-order
   * dimensions are traversed directly using two nested for loops, and the
   * remaining ones via a carry bit. This enables us to obtain efficient
   * low-level code for the two inner-most loops.
   *
   * \param shape the shape of the table that we traverse linearly
   * \param x_ptr the pointer to the first element of table X
   * \param x_inc the pointer increment in X for each dimension in shape
   * \param op    a unary operation accepting x_ptr
   *
   * \return the unary operation in its final state
   */
  template <typename Tx, typename Op>
  Op unary_loop(const uint_vector& shape,
                Tx* x_ptr, table_increment&& x_inc,
                Op op) {
    // initialize the index
    assert(shape.size() < 16);
    std::size_t index[16];
    std::copy(shape.begin(), shape.end(), index);
    index[shape.size()] = 0;

    // the special case with zero dimensions (i.e., a single element)
    if (shape.empty()) {
      op(x_ptr);
      return op;
    }

    // the special case with 1 dimension
    if (shape.size() == 1) {
      std::ptrdiff_t x_inc0 = x_inc[0];
      for (std::size_t i = shape[0]; i; --i) {
        op(x_ptr);
        x_ptr += x_inc0;
      }
      return op;
    }

    // the general case
    x_inc.partial_sum(2);
    std::ptrdiff_t x_inc0 = x_inc[0];
    std::ptrdiff_t x_inc1 = x_inc[1];
    std::size_t d; // the last updated bit (initialized in the loop)
    do {
      for (std::size_t j = shape[1]; j; --j) {
        for (std::size_t i = shape[0]; i; --i) {
          op(x_ptr);
          x_ptr += x_inc0;
        }
        x_ptr += x_inc1;
      }
      d = 2;
      while (__builtin_expect(!--index[d], false)) {
        index[d] = shape[d];
        ++d;
      }
      x_ptr += x_inc[d];
    } while (d != shape.size());
    return op;
  }

  /**
   * A function that loops linearly over one table and non-linearly
   * over two other tables.
   *
   * This function implements a hybrid approach, where the two lowest-order
   * dimensions are traversed directly using two nested for loops, and the
   * remaining ones via a carry bit. This enables us to obtain efficient
   * low-level code for the two inner-most loops.
   *
   * \param shape the shape of the table that we traverse linearly
   * \param x_ptr the pointer to the first element of table X
   * \param x_inc the pointer increment in X for each dimension in shape
   * \param y_ptr the pointer to the first element of table Y
   * \param y_inc the pointer increment in Y for each dimension in shape
   * \param op    a binary operation accepting x_ptr and y_ptr
   *
   * \return the binary operation in its final state
   */
  template <typename Tx, typename Ty, typename Op>
  Op binary_loop(const uint_vector& shape,
                 Tx* x_ptr, table_increment&& x_inc,
                 Ty* y_ptr, table_increment&& y_inc,
                 Op op) {
    // initialize the index
    assert(shape.size() < 16);
    std::size_t index[16];
    std::copy(shape.begin(), shape.end(), index);
    index[shape.size()] = 0;

    // the special case with zero dimensions (i.e., a single element)
    if (shape.empty()) {
      op(x_ptr, y_ptr);
      return op;
    }

    // the special case with 1 dimension
    if (shape.size() == 1) {
      std::ptrdiff_t x_inc0 = x_inc[0];
      std::ptrdiff_t y_inc0 = y_inc[0];
      for (std::size_t i = shape[0]; i; --i) {
        op(x_ptr, y_ptr);
        x_ptr += x_inc0;
        y_ptr += y_inc0;
      }
      return op;
    }

    // the general case
    x_inc.partial_sum(2);
    y_inc.partial_sum(2);
    std::ptrdiff_t x_inc0 = x_inc[0];
    std::ptrdiff_t y_inc0 = y_inc[0];
    std::ptrdiff_t x_inc1 = x_inc[1];
    std::ptrdiff_t y_inc1 = y_inc[1];
    std::size_t d; // the last updated bit (initialized in the loop)
    do {
      for (std::size_t j = shape[1]; j; --j) {
        for (std::size_t i = shape[0]; i; --i) {
          op(x_ptr, y_ptr);
          x_ptr += x_inc0;
          y_ptr += y_inc0;
        }
        x_ptr += x_inc1;
        y_ptr += y_inc1;
      }
      d = 2;
      while (__builtin_expect(!--index[d], false)) {
        index[d] = shape[d];
        ++d;
      }
      x_ptr += x_inc[d];
      y_ptr += y_inc[d];
    } while (d != shape.size());
    return op;
  }

  /**
   * A function that loops linearly over one table and non-linearly
   * over three other tables.
   *
   * This function implements a hybrid approach, where the two lowest-order
   * dimensions are traversed directly using two nested for loops, and the
   * remaining ones via a carry bit. This enables us to obtain efficient
   * low-level code for the two inner-most loops.
   *
   * \param shape the shape of the table that we traverse linearly
   * \param x_ptr the pointer to the first element of table X
   * \param x_inc the pointer increment in X for each dimension in shape
   * \param y_ptr the pointer to the first element of table Y
   * \param y_inc the pointer increment in Y for each dimension in shape
   * \param z_ptr the pointer to the first element of table Z
   * \param z_inc the pointer increment in Z for each dimension in shape
   * \param op    a ternary operation accepting x_ptr, y_ptr, and z_ptr
   *
   * \return op the ternary operation in its final state
   */
  template <typename Tx, typename Ty, typename Tz, typename Op>
  Op ternary_loop(const uint_vector& shape,
                  Tx* x_ptr, table_increment&& x_inc,
                  Ty* y_ptr, table_increment&& y_inc,
                  Tz* z_ptr, table_increment&& z_inc,
                  Op op) {
    // initialize the index
    assert(shape.size() < 16);
    std::size_t index[16];
    std::copy(shape.begin(), shape.end(), index);
    index[shape.size()] = 0;

    // the special case with zero dimensions (i.e., a single element)
    if (shape.empty()) {
      op(x_ptr, y_ptr, z_ptr);
      return op;
    }

    // the special case with 1 dimension
    if (shape.size() == 1) {
      std::ptrdiff_t x_inc0 = x_inc[0];
      std::ptrdiff_t y_inc0 = y_inc[0];
      std::ptrdiff_t z_inc0 = z_inc[0];
      for (std::size_t i = shape[0]; i; --i) {
        op(x_ptr, y_ptr, z_ptr);
        x_ptr += x_inc0;
        y_ptr += y_inc0;
        z_ptr += z_inc0;
      }
      return op;
    }

    // the general case
    x_inc.partial_sum(2);
    y_inc.partial_sum(2);
    z_inc.partial_sum(2);
    std::ptrdiff_t x_inc0 = x_inc[0];
    std::ptrdiff_t y_inc0 = y_inc[0];
    std::ptrdiff_t z_inc0 = z_inc[0];
    std::ptrdiff_t x_inc1 = x_inc[1];
    std::ptrdiff_t y_inc1 = y_inc[1];
    std::ptrdiff_t z_inc1 = z_inc[1];
    std::size_t d; // the last updated bit (initialized in the loop)
    do {
      for (std::size_t j = shape[1]; j; --j) {
        for (std::size_t i = shape[0]; i; --i) {
          op(x_ptr, y_ptr, z_ptr);
          x_ptr += x_inc0;
          y_ptr += y_inc0;
          z_ptr += z_inc0;
        }
        x_ptr += x_inc1;
        y_ptr += y_inc1;
        z_ptr += z_inc1;
      }
      d = 2;
      while (__builtin_expect(!--index[d], false)) {
        index[d] = shape[d];
        ++d;
      }
      x_ptr += x_inc[d];
      y_ptr += y_inc[d];
      z_ptr += z_inc[d];
    } while (d != shape.size());
    return op;
  }


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
  class table {
  public:
    // Public type declarations
    //--------------------------------------------------------------------------

    //! The type of values stored in this object
    typedef T value_type;

    //! An iterator over the elements of this table in a linear fashion
    typedef T* iterator;

    //! A const iterator over the elements of this table in a linear fashion
    typedef const T* const_iterator;

    //! The type used to represent the indices and shape
    typedef uint_vector index_type;

    //! The iterator over all indices in this table
    typedef uint_vector_iterator index_iterator;

    // Constructors and initialization
    //--------------------------------------------------------------------------

    /**
     * Default constructor. Creates an empty table with no elements.
     */
    table() { }

    /**
     * Constructs a table with the given shape. This constructor does not
     * initialize the table elements. If needed, the table contents can be
     * initialized with other constructors or with the the fill() function.
     */
    explicit table(const uint_vector& shape) {
      reset(shape);
    }

    /**
     * Constructs a table with the given shape and initializes its elements
     * to the given value.
     */
    table(const uint_vector& shape, const T& init) {
      reset(shape);
      fill(init);
    }

    /**
     * Constructs a table with the given shape and initializes its elements
     * to the given list.
     */
    table(const uint_vector& shape, std::initializer_list<T> values) {
      reset(shape);
      assert(size() == values.size());
      std::copy(values.begin(), values.end(), data());
    }

    /**
     * Copy constructor. Copies the shape and elements of a table to this one.
     */
    table(const table& x)
      : shape_(x.shape_), offset_(x.offset_) {
      data_.reset(new T[size()]);
      std::copy(x.begin(), x.end(), data());
    }

    //! Move constructor
    table(table&& x) {
      swap(*this, x);
    }

    //! Assignment operator. Copies the shape and elements from x to this table.
    table& operator=(const table& x) {
      if (this != &x) {
        if (size() != x.size()) { data_.reset(new T[x.size()]); }
        shape_ = x.shape_;
        offset_ = x.offset_;
        std::copy(x.begin(), x.end(), data());
      }
      return *this;
    }

    table& operator=(table&& x) {
      swap(*this, x);
      return *this;
    }

    //! Swaps the contents of two tables.
    friend void swap(table& x, table& y) {
      if (&x != &y) {
        using std::swap;
        swap(x.shape_, y.shape_);
        swap(x.offset_, y.offset_);
        swap(x.data_, y.data_);
      }
    }

    //! Saves the table to an archive.
    void save(oarchive& ar) const {
      ar << shape_;
      ar.serialize_range(begin(), end());
    }

    //! Loads the table from an archive.
    void load(iarchive& ar) {
      uint_vector shape;
      ar >> shape;
      reset(shape);
      ar.deserialize_range<T>(data());
    }

    /**
     * Reset the table to the given shape. Allocates the memory, but does
     * not initialize the elements.
     */
    void reset(const uint_vector& shape) {
      std::size_t old_size = size();
      shape_ = shape;
      offset_.reset(shape); // affects size()
      if (size() != old_size) { data_.reset(new T[size()]); }
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (std::numeric_limits<std::size_t>::max() / shape[i] <=
            offset_.multiplier(i)) {
          throw std::out_of_range("table::reset possibly overflows size_t");
        }
      }
    }

    /**
     * Resets the table to the shape given by a range of two iterators.
     * Allocates the memory, but does not initialize the elements.
     */
    template <typename It>
    void reset(It start, It end) {
      std::size_t old_size = size();
      shape_.assign(start, end);
      offset_.reset(shape_); // affects size()
      if (size() != old_size) { data_.reset(new T[size()]); }
      for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (std::numeric_limits<std::size_t>::max() / shape_[i] <=
            offset_.multiplier(i)) {
          throw std::out_of_range("table::reset possibly overflows size_t");
        }
      }
    }

    /**
     * Verifies that the specified shape matches this table.
     */
    void check_shape(const uint_vector& shape) const {
      if (shape_ != shape) {
        throw std::invalid_argument("Incompatible shapes");
      }
    }

    /**
     * Verifies that the specified shape matches this table.
     */
    template <typename It>
    void check_shape(It begin, It end) const {
      if (!std::equal(shape_.begin(), shape_.end(), begin, end)) {
        throw std::invalid_argument("Incompatible shapes");
      }
    }

    // Accessors
    //--------------------------------------------------------------------------

    //! Returns the number of dimensions of this table.
    std::size_t arity() const {
      return shape_.size();
    }

    //! Returns the dimensions of this table.
    const uint_vector& shape() const {
      return shape_;
    }

    //! Returns the offset object for this table.
    const table_offset& offset() const {
      return offset_;
    }

    //! Returns the size of the given dimension of this table
    std::size_t size(std::size_t n) const {
      return shape_[n];
    }

    //! Total number of elements in the table.
    std::size_t size() const {
      return offset_.num_elements();
    }

    //! Returns true if the table was default-initialized, i.e. has no elements.
    bool empty() const {
      return offset_.num_elements() == 0;
    }

    //! Returns a pointer to the allocated values or null if the table is empty.
    T* data() {
      return data_.get();
    }

    //! Returns a pointer to the allocated values or null if the table is empty.
    const T* data() const {
      return data_.get();
    }

    //! Returns the pointer (iterator) to the first element.
    T* begin() {
      return data_.get();
    }

    //! Returns the pointer (iterator) to the first element.
    const T* begin() const {
      return data_.get();
    }

    //! Returns the pointer (iterator) to the one past the last element.
    T* end() {
      return data_.get() + size();
    }

    //! Returns the pointer (iterator) to the one past the last element.
    const T* end() const {
      return data_.get() + size();
    }

    //! Returns the index associated with an iterator position
    uint_vector index(const T* it) const {
      uint_vector result;
      offset_.vector(it - begin(), result);
      return result;
    }

    //! Returns an iterator range over indices into this table.
    iterator_range<index_iterator> indices() const {
      if (empty()) {
        return { index_iterator(), index_iterator() };
      } else {
        return { index_iterator(shape_), index_iterator(arity()) };
      }
    }

    //! Returns the element with the specified linear index
    T& operator[](std::size_t i) {
      return data_[i];
    }

    //! Returns the element with the specified linear index
    const T& operator[](std::size_t i) const {
      return data_[i];
    }

    //! Returns the element with the specified finite index
    T& operator()(const uint_vector& index) {
      return data_[offset_.linear(index)];
    }

    //! Returns the element with the specified finite index
    const T& operator()(const uint_vector& index) const {
      return data_[offset_.linear(index)];
    }

    //! Returns true if the two tables have the same shape and elements
    bool operator==(const table& x) const {
      return shape_ == x.shape_ && std::equal(begin(), end(), x.begin());
    }

    //! Returns true if the two tables do not have the same shape or elements
    bool operator!=(const table& x) const {
      return !(*this == x);
    }

    // Sequential operations
    //--------------------------------------------------------------------------

    /**
     * Sets all the elements to the given value.
     */
    void fill(const T& value) {
      std::fill(begin(), end(), value);
    }

    /**
     * Transforms the content of this table by applying a unary operation
     * to each element.
     */
    template <typename Op>
    table<T>& transform(Op op) {
      for (std::size_t i = 0; i < size(); ++i) {
        data_[i] = op(data_[i]);
      }
      return *this;
    }

    /**
     * A convenience function that does not do anything.
     */
    table<T>& transform(identity) {
      return *this;
    }

    /**
     * Transforms the content of this table by applying a binary operation
     * to the elements of this table and the given table. The two tables
     * must have the same shapes.
     */
    template <typename Op>
    table<T>& transform(const table& x, Op op) {
      assert(shape() == x.shape());
      for (std::size_t i = 0; i < size(); ++i) {
        data_[i] = op(data_[i], x.data_[i]);
      }
      return *this;
    }

    /**
     * Aggregates all the elements of the table using the given operation,
     * initializing the aggregate with the given value.
     */
    template <typename U, typename Op>
    U accumulate(U init, Op op) const {
      return std::accumulate(begin(), end(), init, op);
    }

    /**
     * Transforms all the elements of this table using the given unary
     * operation and accumulates the result.
     */
    template <typename R, typename TransOp, typename AccuOp>
    R transform_accumulate(R init, TransOp trans_op, AccuOp accu_op) const {
      R result(init);
      for (std::size_t i = 0; i < size(); ++i) {
        result = accu_op(result, trans_op(data_[i]));
      }
      return result;
    }

    /**
     * Restricts the given table to an assignment to all the dimensions
     * >= n, where n is the arity of this table. The parameter x_start is
     * the linear index of the first element in the input table that should
     * be copied. This table must be preallocated, and its first n
     * dimensions must match the first n dimensions of the input table.
     */
    void restrict(const table& x, std::size_t x_start) {
      assert(arity() <= x.arity());
      assert(std::equal(shape_.begin(), shape_.end(), x.shape_.begin()));
      assert(x_start + size() <= x.size());
      std::copy(x.data() + x_start, x.data() + x_start + size(), data());
    }

    /**
     * Draws a sample when this table represents a distribution.
     * \param trans_op an object that transforms the elements to probabilities
     * \param rng a random number generator object
     * \param tail the assignment to tail arguments
     */
    template <typename TransOp, typename Generator>
    uint_vector
    sample(TransOp trans, Generator& rng, const uint_vector& tail) const {
      assert(tail.size() < arity());
      std::size_t nhead = arity() - tail.size();
      std::size_t nelem = offset().multiplier(nhead);
      const T* elem = data() + offset().linear(tail, nhead);
      T p = std::uniform_real_distribution<T>()(rng);
      for (std::size_t i = 0; i < nelem; ++i) {
        T prob = trans(elem[i]);
        if (p <= prob) {
          uint_vector result;
          offset().vector(i, nhead, result);
          return result;
        } else {
          p -= prob;
        }
      }
      throw std::invalid_argument("The total probability is less than 1");
    }

    /**
     * Identifies the first elements that satisfies the given predicate
     * and stores the corresponding index to an output vector.
     *
     * \throw std::out_of_range if the element cannot be found.
     */
    template <typename UnaryPredicate>
    void find_if(UnaryPredicate pred, uint_vector& index) const {
      offset_.vector(std::find_if(begin(), end(), pred) - begin(), index);
    }

    // Factor operations
    //--------------------------------------------------------------------------

    /**
     * Joins two tables.
     * The left table is iterated over in a sequential fashion, while the second
     * one is iterated as specified by the mapping dim_map.
     *
     * \param join_op  A binary operation operating on the elements of f and g.
     * \param f        The left table.
     * \param g        The right table.
     * \param g_map    The mapping from the second table to the joint table.
     * \param result   The output table.
     */
    template <typename JoinOp>
    friend void join(JoinOp join_op,
                     const table& f,
                     const table& g,
                     const uint_vector& g_map,
                     table& result) {
      // reset the output table
      auto it = std::max_element(g_map.begin(), g_map.end());
      if (it == g_map.end() || *it < f.arity()) { // left join
        result.reset(f.shape());
      } else {                                    // right or outer join
        result.reset(join_shape(*it + 1, f.shape(), g.shape(), g_map));
      }

      // iterate linearly over the output and non-linearly over the inputs
      T* r = result.data();
      binary_loop(result.shape(),
                  f.data(), table_increment(result.shape(), f.offset()),
                  g.data(), table_increment(result.shape(), g.offset(), g_map),
                  [join_op, r] (const T* x, const T* y) mutable {
                    *r++ = join_op(*x, *y);
                  });
    }

    /**
     * Joins this table into another one inplace.
     *
     * \param join_op  A binary operation operating on the elements of f and g.
     * \param dim_map  A mapping from this table to the result.
     * \param result   The output table.
     */
    template <typename JoinOp>
    void join_inplace(JoinOp join_op, const uint_vector& dim_map,
                      table& result) const {
      // check that dimensions match
      assert(arity() == dim_map.size());
      for (std::size_t i = 0; i < dim_map.size(); ++i) {
        assert(size(i) == result.size(dim_map[i]));
      }

      // iterate linearly over the output and non-linearly over the input
      T* r = result.data();
      unary_loop(result.shape(),
                 data(), table_increment(result.shape(), offset_, dim_map),
                 [join_op, r] (const T* x) mutable {
                   *r = join_op(*r, *x);
                   ++r;
                 });
    }

    /**
     * Joins two tables and accumulates the result of the join.
     * The left table is iterated over in a sequential fashion, while the second
     * one is iterated as specified by the mapping dim_map.
     *
     * \param join_op  A binary operation operating on the elements of f and g.
     * \param agg_op   A binary aggregation operation.
     * \param init     The initial value for the aggregate.
     * \param f        The left table.
     * \param g        The right table.
     * \param g_map    The mapping from the second table to the joint table.
     */
    template <typename JoinOp, typename AggOp>
    friend T join_accumulate(JoinOp join_op,
                             AggOp agg_op,
                             T init,
                             const table& f,
                             const table& g,
                             const uint_vector& g_map) {
      // compute the shape of the output table
      uint_vector shape;
      auto it = std::max_element(g_map.begin(), g_map.end());
      if (it == g_map.end() || *it < f.arity()) { // left join
        shape = f.shape();
      } else {                                    // right or outer join
        shape = join_shape(*it + 1, f.shape(), g.shape(), g_map);
      }

      // iterate non-linearly over the inputs, aggregating the result
      return binary_loop(
        shape,
        f.data(), table_increment(shape, f.offset()),
        g.data(), table_increment(shape, g.offset(), g_map),
        join_accu<JoinOp, AggOp>{join_op, agg_op, init}
      ).result;
    }

    //! A helper for join_accumulate
    template <typename JoinOp, typename AggOp>
    struct join_accu {
      JoinOp join_op;
      AggOp agg_op;
      T result;
      void operator()(const T* x, const T* y) {
        result = agg_op(result, join_op(*x, *y));
      }
    };

    /**
     * Aggregates this table using a binary aggregate operation,
     * retaining the specified dimensions.
     *
     * \param agg_op   A binary aggregation operation.
     * \param init     The initial value for the aggregate.
     * \param retain   The retained dimensions.
     * \param result   The output table.
     */
    template <typename AggOp>
    void aggregate(AggOp agg_op,
                   T init,
                   const uint_vector& retain,
                   table& result) const {
      // reset the output table
      result.reset(aggregate_shape(shape_, retain));
      result.fill(init);

      // iterate linearly over the input and non-linearly over the output
      const T* in = data();
      unary_loop(shape_,
                 result.data(), table_increment(shape_, result.offset(), retain),
                 [agg_op, in] (T* r) mutable {
                   *r = agg_op(*r, *in++);
                 });
    }

    /**
     * A specialization of aggregate for the log-sum-exp operation.
     */
    void aggregate(log_plus_exp<T> /* ignored */,
                   T /* ignored */,
                   const uint_vector& retain,
                   table& result) const {
      // reset the output table
      result.reset(aggregate_shape(shape_, retain));
      result.fill(T(0));

      // iterate linearly over the input and non-linearly over the output
      const T* in = data();
      T offset = *std::max_element(begin(), end());
      unary_loop(shape_,
                 result.data(), table_increment(shape_, result.offset(), retain),
                 [in, offset] (T* r) mutable {
                   *r += std::exp(*in++ - offset);
                 });
      for (T& x : result) { x = std::log(x) + offset; }
    }

    /**
     * Aggregates this table using a binary aggregate operation,
     * retaining the specified dimension (not optimized at the moment).
     *
     * \param agg_op   A binary aggregation operation.
     * \param init     The initial value for the aggregate.
     * \param retain   The retained dimension.
     * \param result   The output table.
     */
    template <typename AggOp>
    void aggregate(AggOp agg_op,
                   T init,
                   std::size_t retain,
                   table& result) const {
      aggregate(agg_op, init, uint_vector({retain}), result);
    }

    /**
     * Eliminates the specified dimensions from this table using a binary
     * operation.
     *
     * \param agg_op   A binary aggregation operation.
     * \param init     The initial value for the aggregate.
     * \param dims     The eliminated dimensions.
     * \param result   The output table.
     */
    template <typename AggOp>
    void eliminate(AggOp agg_op,
                   T init,
                   const uint_vector& dims,
                   table& result) const {
      aggregate(agg_op, init, retained_dims(arity(), dims), result);
    }

    /**
     * Joins two tables and aggregates the result into an output table.
     * The left table is iterated over in a sequential fashion, while the second
     * one is iterated as specified by the mapping dim_map.
     *
     * \param join_op  A binary operation operating on the elements of f and g.
     * \param agg_op   A binary aggregation operation.
     * \param init     The initial value for the aggregate.
     * \param f        The left table.
     * \param g        The right table.
     * \param g_map    The mapping from the second table to the joint table.
     * \param retain   The retained dimensions of the ficticious table.
     * \param result   The output table.
     */
    template <typename JoinOp, typename AggOp>
    friend void join_aggregate(JoinOp join_op,
                               AggOp agg_op,
                               T init,
                               const table& f,
                               const table& g,
                               const uint_vector& g_map,
                               const uint_vector& retain,
                               table& result) {
      // reset the output table
      auto it = std::max_element(g_map.begin(), g_map.end());
      uint_vector shape;
      if (it == g_map.end() || *it < f.arity()) { // left join
        shape = f.shape();
      } else {                                    // right or outer join
        shape = join_shape(*it + 1, f.shape(), g.shape(), g_map);
      }
      result.reset(aggregate_shape(shape, retain));
      result.fill(init);

      // iterate linearly over the ficticious table and non-linearly over the
      // inputs and the output
      ternary_loop(shape,
                   f.data(), table_increment(shape, f.offset()),
                   g.data(), table_increment(shape, g.offset(), g_map),
                   result.data(), table_increment(shape, result.offset(), retain),
                   [join_op, agg_op] (const T* x, const T* y, T* r) {
                     *r = agg_op(*r, join_op(*x, *y));
                   });
    }

    /**
     * Specialization of join_aggregate for the log-sum-exp aggregation.
     */
    template <typename JoinOp>
    friend void join_aggregate(JoinOp join_op,
                               log_plus_exp<T> agg_op,
                               T init,
                               const table& f,
                               const table& g,
                               const uint_vector& g_map,
                               const uint_vector& retain,
                               table& result) {
      table tmp;
      join(join_op, f, g, g_map, tmp);
      tmp.aggregate(agg_op, init, retain, result);
    }

    /**
     * Joins two tables and aggregates the result into an output table.
     * The left table is iterated over in a sequential fashion, while the second
     * one is iterated as specified by the mapping dim_map.
     */
    template <typename JoinOp, typename AggOp>
    friend void join_aggregate(JoinOp join_op,
                               AggOp agg_op,
                               T init,
                               const table& f,
                               const table& g,
                               const uint_vector& g_map,
                               std::size_t retain,
                               table& result) {
      join_aggregate(join_op, agg_op, init, f, g, g_map, uint_vector({retain}),
                     result);
    }

    /**
     * Restricts the head dimensions of this table, retaining the remaining
     * tail dimensions.
     *
     * \param values  The assignment to the head dimensions.
     * \param result  The output table.
     */
    void restrict_head(const uint_vector& values, table& result) const {
      // reset the output table
      assert(values.size() <= arity());
      result.reset(shape_.begin(), shape_.end() - values.size());

      // direct copy
      std::size_t start = offset_.linear(values, result.arity());
      std::copy(data() + start, data() + start + result.size(), result.data());
    }

    /**
     * Restricts the head dimensions of this table and updates the output
     * table using a binary operation.
     *
     * \param values  The assignment to the head dimensions.
     * \param join_op A binary operation accepting the elements of the output
     *                and this table.
     * \param result  The output table.
     */
    template <typename JoinOp>
    void restrict_head_update(const uint_vector& values,
                              JoinOp join_op,
                              table& result) const {
      // check the shape of the output table
      assert(values.size() <= arity());
      result.check_shape(shape_.begin(), shape_.end() - values.size());

      // direct transform
      std::size_t start = offset_.linear(values, result.arity());
      std::transform(result.begin(), result.end(), data() + start,
                     result.begin(), join_op);
    }

    /**
     * Restricts the head dimensions and computes the index of the first
     * element of the result that satisfies the given predicate.
     */
    template <typename UnaryPredicate>
    void restrict_head_find_if(const uint_vector& values,
                               UnaryPredicate pred,
                               uint_vector& result) const {
      assert(values.size() <= arity());
      std::size_t ntail = arity() - values.size();
      const T* begin = begin() + offset_.linear(values, ntail);
      const T* end = begin + offset_.multiplier(ntail);
      offset_.vector(std::find_if(begin, end) - begin, ntail, result);
    }

    /**
     * Restricts the tail dimensions of this table, retaining the remaining
     * head dimensions.
     *
     * \param values  The assignment to the tail dimensions.
     * \param result  The output table.
     */
    void restrict_tail(const uint_vector& values, table& result) const {
      // reset the output table
      assert(values.size() <= arity());
      result.reset(shape_.end() - values.size(), shape_.end());

      // skip copy
      std::size_t linear = offset_.linear(values, 0);
      std::size_t stride = size() / result.size();
      for (std::size_t i = 0; i < result.size(); ++i, linear += stride) {
        result[i] = data_[linear];
      }
    }

    /**
     * Restricts the tail dimensions of this table and updates the output
     * table using a binary operation.
     *
     * \param values  The assignment to the tail dimensions.
     * \param join_op A binary operation accepting the elements of the output
     *                and this table.
     * \param result  The output table.
     */
    template <typename JoinOp>
    void restrict_tail_update(const uint_vector& values,
                              JoinOp join_op,
                              table& result) const {
      // reset the output table
      assert(values.size() <= arity());
      result.check_shape(shape_.end() - values.size(), shape_.end());

      // skip update
      std::size_t linear = offset_.linear(values, 0);
      std::size_t stride = size() / result.size();
      for (std::size_t i = 0; i < result.size(); ++i, linear += stride) {
        result[i] = join_op(result[i], data_[linear]);
      }
    }

    /**
     * Restricts the tail dimensions and computes the index of the first
     * element of the result that satisfies the given predicate.
     */
    template <typename UnaryPredicate>
    void restrict_tail_find_if(const uint_vector& values,
                               UnaryPredicate pred,
                               uint_vector& result) const {
      assert(values.size() <= arity());
      std::size_t stride = offset_.multiplier(values.size());
      std::size_t linear = offset_.linear(values, 0);
      while (linear < size() && !pred(data_[linear])) {
        linear += stride;
      }
      offset_.vector(linear, values.size(), arity(), result);
    }

    /**
     * Restricts the specified dimensions of this table to the given values,
     * retaining the remaining dimensions.
     *
     * \param dims    The restricted dimensions.
     * \param values  The assignment to the restricted dimensions.
     * \param result  The output table.
     */
    void restrict(const uint_vector& dims,
                  const uint_vector& values,
                  table& result) const {
      assert(dims.size() == values.size());

      // reset the output table
      uint_vector dim_map = restrict_map(dims);
      uint_vector shape = restrict_shape(arity() - dims.size(), dim_map);
      result.reset(shape);

      // iterate linearly over the output and non-linearly over the input
      T* r = result.data();
      std::size_t start = offset_.linear(dims, values);
      unary_loop(result.shape(),
                 data() + start, table_increment(result.shape(), offset_, dim_map),
                 [r] (const T* x) mutable {
                   *r++ = *x;
                 });
    }

    /**
     * Restricts the specified dimensions of this table to the given values
     * and updates the output table using the given binary operation.
     *
     * \param dims    The restricted dimensions.
     * \param values  The assignment to the restricted dimensions.
     * \param join_op A binary operation accepting the elements of the output
     *                and this table.
     * \param result  The output table.
     */
    template <typename JoinOp>
    void restrict_update(const uint_vector& dims,
                         const uint_vector& values,
                         JoinOp join_op,
                         table& result) const {
      assert(dims.size() == values.size());

      // reset the output table
      uint_vector dim_map = restrict_map(dims);
      uint_vector shape = restrict_shape(arity() - dims.size(), dim_map);
      result.check_shape(shape);

      // iterate linearly over the output and non-linearly over the input
      T* r = result.data();
      std::size_t start = offset_.linear(dims, values);
      unary_loop(result.shape(),
                 data() + start, table_increment(result.shape(), offset_, dim_map),
                 [r, join_op] (const T* x) mutable {
                   *r = join_op(*r, *x);
                   ++r;
                 });
    }

    /**
     * Restricts the specified dimensions of this table to the given values
     * and joins the result into the output table using a binary operation.
     *
     * \param dims    The restricted dimensions.
     * \param values  The assignment to the restricted dimensions.
     * \param r_map   A map from the restricted result to the output table.
     * \param join_op A binary operation accepting the elements of the output
     *                and this table.
     * \param result  The output table.
     */
    template <typename JoinOp>
    void restrict_join(const uint_vector& dims,
                       const uint_vector& values,
                       const uint_vector& r_map,
                       JoinOp join_op,
                       table& result) const {
      assert(dims.size() == values.size());

      // reset the output table
      uint_vector dim_map = restrict_map(dims, r_map);
      // todo: check shape

      // iterate linearly over the output and non-linearly over the input
      T* r = result.data();
      std::size_t start = offset_.linear(dims, values);
      unary_loop(result.shape(),
                 data() + start, table_increment(result.shape(), offset_, dim_map),
                 [r, join_op] (const T* x) mutable {
                   *r = join_op(*r, *x);
                   ++r;
                 });
    }

    /**
     * Reorders the dimensions of this table and stores the result
     * to an output table.
     *
     * \param order  The ordering of indices in the output table, i.e.,
     *               mapping from the output table dimensions to this one.
     * \param result The result table.
     */
    void reorder(const uint_vector& order, table& result) const {
      assert(order.size() == arity());
      result.reset(aggregate_shape(shape_, order));

      // iterate linearly over the output and non-linearly over the input
      const T* x = data();
      unary_loop(shape_,
                 result.data(), table_increment(shape_, result.offset(), order),
                 [x] (T* r) mutable {
                   *r = *x++;
                 });
    }

  private:
    static uint_vector join_shape(std::size_t n,
                                  const uint_vector& f_shape,
                                  const uint_vector& g_shape,
                                  const uint_vector& g_map) {
      uint_vector shape(n, missing<std::size_t>::value);
      std::copy(f_shape.begin(), f_shape.end(), shape.begin());
      for (std::size_t i = 0; i < g_map.size(); ++i) {
        std::size_t& size = shape[g_map[i]];
        if (size == missing<std::size_t>::value) {
          size = g_shape[i];
        } else {
          assert(size == g_shape[i]);
        }
      }
      return shape;
    }

    static uint_vector aggregate_shape(const uint_vector& f_shape,
                                       const uint_vector& retain) {
      uint_vector shape(retain.size());
      for (std::size_t i = 0; i < retain.size(); ++i) {
        shape[i] = f_shape[retain[i]];
      }
      return shape;
    }

    uint_vector restrict_map(const uint_vector& dims) const {
      uint_vector dim_map(arity(), 0);
      for (std::size_t d : dims) {
        dim_map[d] = missing<std::size_t>::value;
      }
      std::size_t i = 0;
      for (std::size_t& d : dim_map) {
        if (d == 0) { d = i++; }
      }
      assert(i == arity() - dims.size()); // check for duplicates
      return dim_map;
    }

    uint_vector restrict_map(const uint_vector& dims,
                             const uint_vector& r_map) const {
      assert(arity() - dims.size() == r_map.size());
      uint_vector dim_map(arity(), 0);
      for (std::size_t d : dims) {
        dim_map[d] = missing<std::size_t>::value;
      }
      std::size_t i = 0;
      for (std::size_t& d : dim_map) {
        if (d == 0) { d = r_map[i++]; }
      }
      assert(i == r_map.size()); // check for duplicates
      return dim_map;
    }

    uint_vector restrict_shape(std::size_t n, const uint_vector& map) const {
      uint_vector shape(n);
      for (std::size_t i = 0; i < map.size(); ++i) {
        if (map[i] != missing<std::size_t>::value) {
          shape[map[i]] = shape_[i];
        }
      }
      return shape;
    }

    // Private members
    //--------------------------------------------------------------------------

    //! The dimensions of this table.
    uint_vector shape_;

    //! Translates between finite and linear indices.
    table_offset offset_;

    //! The elements of this table, stored in a linear order.
    std::unique_ptr<T[]> data_;

  }; // class table

  /**
   * Prints a human-readable representation of the table to an output stream.
   * \relates table
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const table<T>& table) {
    std::ostream_iterator<std::size_t, char> out_it(out, " ");
    const T* elem = table.data();
    for (const uint_vector& index :  table.indices()) {
      std::copy(index.begin(), index.end(), out_it);
      out << *elem++ << std::endl;
    }
    return out;
  }

  // OptimizationVector functions
  //==========================================================================

  /**
   * Adds the elements of table x to the elements of table y in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator+=(table<T>& y, const table<T>& x) {
    return y.transform(x, std::plus<T>());
  }

  /**
   * Subtracts the elements of table x from the elements of table y in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator-=(table<T>& y, const table<T>& x) {
    return y.transform(x, std::minus<T>());
  }

  /**
   * Adds a constant to all elements of table x in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator+=(table<T>& x, const T& a) {
    return x.transform(incremented_by<T>(a));
  }

  /**
   * Subtracts a constant from all elements of table x in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator-=(table<T>& x, const T& a) {
    return x.transform(decremented_by<T>(a));
  }

  /**
   * Multiplies all elements of table x by a constant in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator*=(table<T>& x, const T& a) {
    return x.transform(multiplied_by<T>(a));
  }

  /**
   * Divides all elements table x by a constant in place.
   * \relates table
   */
  template <typename T>
  table<T>& operator/=(table<T>& x, const T& a) {
    return x.transform(divided_by<T>(a));
  }

  /**
   * Performs the operation y += a * x.
   * \relates table
   */
  template <typename T>
  void axpy(const T& a, const table<T>& x, table<T>& y) {
    y.transform(x, plus_multiple<T>(a));
  }

  /**
   * Computes the dot product of two matrices for elementary type T.
   * \relates table
   */
  template <typename T>
  T dot(const table<T>& x, const table<T>& y) {
    assert(x.shape() == y.shape());
    return std::inner_product(x.begin(), x.end(), y.begin(), T(0));
  }

  // Table operations
  //==========================================================================

  template <int N>
  using int_ = std::integral_constant<int, N>;

  /**
   * A utility function that invokes the loop template for the table operation
   * class and the given table arity. The goal is to inline the loops if arity
   * is sufficiently small. Specifically, this function invokes the member
   * function template loop<D> with D=arity if arity is small; otherwise,
   * it throws an error.
   */
  template <typename TableOp>
  void invoke_loop_template(TableOp& table_op, std::size_t arity) {
    switch(arity) {
    case 0: table_op.loop(int_<0>()); break;
    case 1: table_op.loop(int_<1>()); break;
    case 2: table_op.loop(int_<2>()); break;
    case 3: table_op.loop(int_<3>()); break;
    case 4: table_op.loop(int_<4>()); break;
    case 5: table_op.loop(int_<5>()); break;
    case 6: table_op.loop(int_<6>()); break;
    case 7: table_op.loop(int_<7>()); break;
    case 8: table_op.loop(int_<8>()); break;
    case 9: table_op.loop(int_<9>()); break;
    case 10: table_op.loop(int_<10>()); break;
    default: table_op.loop(); break;
    }
  }

  /**
   * Performs a transform operation on one or more tables, storing the result
   * to another table. The input tables must have the same shapes.
   *
   * \tparam T the value type of the resulting table
   * \tparam Op the transform operation that returns value convertible to T
   */
  template <typename T, typename Op>
  class table_transform_assign {
  public:
    table_transform_assign(table<T>& result, Op op)
      : result_(result), op_(op) { }

    template <typename... Ts>
    void operator()(const table<Ts>&... input) const {
      constexpr std::size_t N = sizeof...(Ts);
      if (homogeneous_tuple<const uint_vector&, N>(input.shape()...) !=
          tuple_rep<N>(nth_value<0>(input...).shape())) {
        throw std::invalid_argument("table_transform:: Incompatible shapes");
      }

      std::tuple<const Ts*...> src(input.data()...);
      std::size_t size = nth_value<0>(input...).size();
      result_.reset(nth_value<0>(input...).shape());
      T* dest = result_.data();
      for (std::size_t i = 0; i < size; ++i) {
        *dest++ = tuple_apply(op_, tuple_transform(dereference(), src));
        tuple_transform(preincrement(), src);
      }
    }

  private:
    table<T>& result_;
    Op op_;
  };

  /**
   * Performs a transform operation on one or more tables and updates the
   * the result table with result of this operation elementwise.
   *
   * \tparam T the value type of the result table
   * \tparam TransOp the transform operation that returns value convertible to T
   * \tapram Combine the combination operation
   */
  template <typename T, typename TransOp, typename Combine>
  class table_transform_update {
  public:
    table_transform_update(table<T>& result, TransOp op)
      : result_(result), op_(op) { }

    template <typename... Ts>
    void operator()(const table<Ts>&... input) const {
      assert(result_.shape() == nth_value<0>(input...).shape());
      std::tuple<const Ts*...> src(input.data()...);
      T* r = result_.data();
      std::size_t size = result_.size();
      for (std::size_t i = 0; i < size; ++i) {
        *r = combine_(*r, tuple_apply(op_, tuple_transform(dereference(), src)));
        ++r;
        tuple_transform(preincrement(), src);
      }
    }

  private:
    table<T>& result_;
    TransOp op_;
    Combine combine_;
  };

  /**
   * Follows a transform on one or more tables and accumulates the result.
   *
   * \tparam T the type representing the result
   * \tparam AggOp a binary operation that accumulates the result
   * \tparam TransOp the transform operation
   */
  template <typename T, typename TransOp, typename AggOp>
  class table_transform_accumulate {
  public:
    table_transform_accumulate(T init, TransOp trans_op, AggOp agg_op)
      : init_(init), trans_op_(trans_op), agg_op_(agg_op) { }

    template <typename... Ts>
    T operator()(const table<Ts>&... input) const {
      // TODO: check compatibility
      std::tuple<const Ts*...> ptr(input.data()...);
      std::size_t size = nth_value<0>(input...).size();
      T r = init_;
      for (std::size_t i = 0; i < size; ++i) {
        r = agg_op_(r, tuple_apply(trans_op_, tuple_transform(dereference(), ptr)));
        tuple_transform(preincrement(), ptr);
      }
      return r;
    }

  private:
    T init_;
    TransOp trans_op_;
    AggOp agg_op_;
  };

  /**
   * Performs a join operation on two tables, storing the result to a
   * third table that includes both input tables' dimensions. Each
   * dimension of an input must correspond to exactly one dimension
   * of the result; this mapping is specified by the x_map and y_map
   * indices, where dimension i of x corresponds to the dimension
   * x_map[i] of the result (and similarly for y).
   *
   * The constructor of this class sets up the internal members.
   * To invoke the join operation, use operator().
   *
   * \tparam T the value type of the left table and result
   * \tparam U the value type of the right table
   * \tparam Op a binary operation compatible with T(const T&, const U&)
   */
  template <typename T, typename U, typename Op>
  class table_join {
  public:
    //! Sets up the internal members of the join operation
    table_join(table<T>& result,
               const table<T>& x,
               const table<U>& y,
               const uint_vector& x_map,
               const uint_vector& y_map,
               Op op)
      : shape_(result.shape()),
        r_(result.data()),
        x_(x.data()),
        y_(y.data()),
        x_inc_(result.shape(), x.offset(), x_map),
        y_inc_(result.shape(), y.offset(), y_map),
        op_(op) { }

    //! Performs the join operation, automatically selecting the best method.
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs the join operation for a fixed arity of the result.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        x_ += x_inc_[D-1];
        y_ += y_inc_[D-1];
      }
    }

    //! Performs the join operation for a nullary result (the base case).
    void loop(int_<0>) {
      *r_++ = op_(*x_, *y_);
    };

    //! Performs the join operation for an arbitrary result (slow).
    void loop() {
      x_inc_.partial_sum();
      y_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_++ = op_(*x_, *y_);
        ++it;
        x_ += x_inc_[it.digit()];
        y_ += y_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    const U* y_;
    table_increment x_inc_;
    table_increment y_inc_;
    Op op_;

  }; // class table_join


  /**
   * Joins one table with another. Each dimension of the x table must
   * correspond to a unique dimension of the result table; this
   * mapping is specified by the x_map index, where dimension i of x
   * corresponds to the dimension x_map[i] of the result.
   * This operation updates each element of the result table using
   * the given binary operation, such as std::multiplies.
   *
   * The constructor of this class sets up the internal members.
   * To invoke the join operation, use operator().
   *
   * \tparam T the value type of the result table
   * \tparam U the value type of the input table
   * \tparam Op a binary operation compatible with T(T&, const U&)
   */
  template <typename T, typename U, typename Op>
  class table_join_inplace {
  public:
    //! Sets up the internal members of the join operation
    table_join_inplace(table<T>& result,
                       const table<U>& x,
                       const uint_vector& x_map,
                       Op op)
      : shape_(result.shape()),
        r_(result.data()),
        x_(x.data()),
        x_inc_(result.shape(), x.offset(), x_map),
        op_(op) { }

    //! Performs the join operation, automatically selecting the best method.
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs the join operation for a fixed arity of the result.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        x_ += x_inc_[D-1];
      }
    }

    //! Performs the join operation for a nullary result (the base case).
    void loop(int_<0>) {
      *r_ = op_(*r_, *x_);
      ++r_;
    };

    //! Performs the join operation for an arbitrary result (slow).
    void loop() {
      x_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_ = op_(*r_, *x_);
        ++r_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    table_increment x_inc_;
    Op op_;

  }; // class table_join_inplace


  /**
   * Performs a join operation on two tables, accumulating the resulting
   * (ficticious) table. Each dimension of an input must correspond to exactly
   * one dimension of the result; this mapping is specified by the x_map and
   * y_map indices, where dimension i of x corresponds to the dimension
   * x_map[i] of the result (and similarly for y).
   *
   * The constructor of this class sets up the internal members.
   * To invoke the join operation, use operator().
   *
   * \tparam T the value type of the left table and result
   * \tparam U the value type of the right table
   * \tparam AggOp a binary operation compatible with T(const&, const T&)
   * \tparam JoinOp a binary operation compatible with T(const T&, const U&)
   */
  template <typename T, typename U, typename JoinOp, typename AggOp>
  class table_join_accumulate {
  public:
    //! Sets up the internal members of the join operation
    table_join_accumulate(T init,
                          const table<T>& x,
                          const table<U>& y,
                          const uint_vector& x_map,
                          const uint_vector& y_map,
                          const uint_vector& r_shape,
                          JoinOp join_op,
                          AggOp agg_op)
      : result_(init),
        shape_(r_shape),
        x_(x.data()),
        y_(y.data()),
        x_inc_(r_shape, x.offset(), x_map),
        y_inc_(r_shape, y.offset(), y_map),
        join_op_(join_op),
        agg_op_(agg_op) { }

    //! Performs the join operation, automatically selecting the best method.
    T operator()() {
      invoke_loop_template(*this, shape_.size());
      return result_;
    }

    //! Performs the join operation for a fixed arity of the result.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        x_ += x_inc_[D-1];
        y_ += y_inc_[D-1];
      }
    }

    //! Performs the join operation for a nullary result (the base case).
    void loop(int_<0>) {
      result_ = agg_op_(result_, join_op_(*x_, *y_));
    };

    //! Performs the join operation for an arbitrary result (slow).
    void loop() {
      x_inc_.partial_sum();
      y_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        result_ = agg_op_(result_, join_op_(*x_, *y_));
        ++it;
        x_ += x_inc_[it.digit()];
        y_ += y_inc_[it.digit()];
      }
    }

  private:
    T result_;
    const uint_vector& shape_;
    const T* x_;
    const U* y_;
    table_increment x_inc_;
    table_increment y_inc_;
    JoinOp join_op_;
    AggOp agg_op_;

  }; // class table_join_accumulate


  /**
   * Aggregates a table and stores the result to another table.
   * Each dimension of the result table must correspond to a unique
   * dimension of the input table. The remaining dimensions of the
   * input table are aggregated over. The mapping between the input
   * table and the result table is specified by the dim_map index,
   * where dimension i of in the result corresponds to dimension
   * dim_map[i] of the input. The result table must be initialized
   * to the initial values of the aggregate (often 0).
   *
   * The constructor of this class sets up the internal members.
   * To invoke the aggregate operation, use operator().
   *
   * \tparam T the value type of the result
   * \tparam U the value type of the input
   * \tparam Op a binary operation compatible with T(T&, const U&).
   */
  template <typename T, typename U, typename Op>
  class table_aggregate {
  public:
    //! Sets up the internal members of the aggregate operation
    table_aggregate(table<T>& result,
                    const table<T>& input,
                    const uint_vector& result_map,
                    Op op)
      : shape_(input.shape()),
        r_(result.data()),
        x_(input.data()),
        r_inc_(input.shape(), result.offset(), result_map),
        op_(op) { }

    //! Performs an aggregate operation, automatically selecting the best method
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs an aggregate operation for a fixed arity of the input.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        r_ += r_inc_[D-1];
      }
    }

    //! Performs the aggregate operation for a nullary result (the base case).
    void loop(int_<0>) {
      *r_ = op_(*r_, *x_++);
    };

    //! Performs the aggregate operation for an arbitrary input (slow).
    void loop() {
      r_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_ = op_(*r_, *x_++);
        ++it;
        r_ += r_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    table_increment r_inc_;
    Op op_;

  }; // class table_aggregate


  /**
   * Joins two tables x and y and aggregates the elements of the result
   * into a third table. This operation can be used to implement a
   * binary sum-product operation in one step without an intermediate,
   * temporary table z. The result table needs to be initialized to
   * the initial values of the aggregate (often 0).
   *
   * The caller needs to provide the shape of the (ficticious) table z,
   * as well as the mapping from the dimensions of x, y, and the result
   * to the dimensions of z. This mapping is specified by x_map, y_map
   * and result_map indices, where dimension i of x corresponds to the
   * dimension x_map[i] of z, and similarly for y_map and result_map,
   * respectively.
   *
   * The constructor of this class sets up the internal members.
   * To invoke the join-aggregate operation, use operator().
   *
   * \tparam T the type of all the table values
   * \tparam JoinOp the binary operation compatible with T(const T&, const T&)
   * \tparam AggOp a binary operation compatible with T(T&, const T&).
   */
  template <typename T, typename JoinOp, typename AggOp>
  class table_join_aggregate {
  public:
    //! Sets up the internal members of the join-aggregate operation
    table_join_aggregate(table<T>& result,
                         const table<T>& x,
                         const table<T>& y,
                         const uint_vector& result_map,
                         const uint_vector& x_map,
                         const uint_vector& y_map,
                         const uint_vector& z_shape,
                         JoinOp join_op,
                         AggOp agg_op)
      : shape_(z_shape),
        r_(result.data()),
        x_(x.data()),
        y_(y.data()),
        r_inc_(z_shape, result.offset(), result_map),
        x_inc_(z_shape, x.offset(), x_map),
        y_inc_(z_shape, y.offset(), y_map),
        join_op_(join_op),
        agg_op_(agg_op) { }

    //! Performs the join-aggregate operation, automatically selecting
    //! the best method.
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs the join-aggregate operation for a fixed arity of the z table.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        r_ += r_inc_[D-1];
        x_ += x_inc_[D-1];
        y_ += y_inc_[D-1];
      }
    }

    //! Performs the join-aggregate operation for a nullary z (the base case).
    void loop(int_<0>) {
      *r_ = agg_op_(*r_, join_op_(*x_, *y_));
    };

    //! Performs the join-aggregate operation for an arbitrary z (slow).
    void loop() {
      r_inc_.partial_sum();
      x_inc_.partial_sum();
      y_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_ = agg_op_(*r_, join_op_(*x_, *y_));
        ++it;
        r_ += r_inc_[it.digit()];
        x_ += x_inc_[it.digit()];
        y_ += y_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    const T* y_;
    table_increment r_inc_;
    table_increment x_inc_;
    table_increment y_inc_;
    JoinOp join_op_;
    AggOp agg_op_;

  }; // class table_join_aggregate


  /**
   * Restricts the table to the assignment for a subset of the columns.
   * This operation returns all the elements of the input table, whose
   * restricted columns are equal to some index. The restricted columns,
   * as well as the mapping from the input table to the result table,
   * are specified using the x_map index. If x_map[i] is equal to
   * missing<std::size_t>::value, then dimension i of x is restricted
   * (i.e., missing in the output). Otherwise, dimension i of x corresponds to
   * dimension x_map[i] of the result. The argument first specifies the linear
   * index (in the input table x) that corresponds to the first copied
   * value (i.e., value where all non-restricted columns are 0).
   *
   * The constructor of this class sets up the internal members.
   * To invoke the restrict operation, use operator().
   *
   * \tparam T the value type of the input and result tables
   */
  template <typename T>
  class table_restrict {
  public:
    //! Sets up the internal members of the restrict operation
    table_restrict(table<T>& result,
                   const table<T>& x,
                   const uint_vector& x_map,
                   std::size_t x_first)
      : shape_(result.shape()),
        r_(result.data()),
        x_(x.data() + x_first),
        x_inc_(result.shape(), x.offset(), x_map) { }

    //! Performs a restrict operation, automatically selecting the best method.
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs a restrict operation for a fixed arity of the result.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        x_ += x_inc_[D-1];
      }
    }

    //! Performs the restrict operation for a nullary result (the base case).
    void loop(int_<0>) {
      *r_++ = *x_;
    };

    //! Performs the restrict operation for an arbitrary result (slow).
    void loop() {
      x_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_++ = *x_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    table_increment x_inc_;

  }; // class table_restrict


  /**
   * Restricts the table to the assignment for a subset of the columns
   * This operation takes all the elements of the input table, whose
   * restricted columns are equal to some index, and joins them with
   * the result table using the given binary operation. The restricted
   * columns, as well as the mapping from the input table to the result
   * table, are specified using the x_map index. If x_map[i] is equal to
   * mising<std::size_t>::value, then dimension i of x is restricted
   * (i.e., missing in the output). Otherwise, dimension i of x corresponds to
   * dimension x_map[i] of the result. The argument first specifies the linear
   * index (in the input table x) that corresponds to the first copied
   * value (i.e., value where all non-restricted columns are 0).
   *
   * The constructor of this class sets up the internal members.
   * To invoke the restrict operation, use operator().
   *
   * \tparam T the value type of the result table
   * \tparam U the value type of the input table
   * \tparam JoinOp a binary operation compatible with T(T&, const U&)
   */
  template <typename T, typename U, typename JoinOp>
  class table_restrict_join {
  public:
    //! Sets up the internal members of the restrict-join operation
    table_restrict_join(table<T>& result,
                        const table<T>& x,
                        const uint_vector& x_map,
                        std::size_t first,
                        JoinOp join_op)
      : shape_(result.shape()),
        r_(result.data()),
        x_(x.data() + first),
        x_inc_(result.shape(), x.offset(), x_map),
        join_op_(join_op) { }

    //! Performs the restrict-join operation, automatically selecting
    //! the best method.
    void operator()() {
      invoke_loop_template(*this, shape_.size());
    }

    //! Performs the restrict-join operation for a fixed arity of the result.
    template <int D>
    void loop(int_<D>) {
      for (std::size_t i = shape_[D-1]; i; --i) {
        loop(int_<D-1>());
        x_ += x_inc_[D-1];
      }
    }

    //! Performs a restrict-join operation for a nullary result (the base case).
    void loop(int_<0>) {
      *r_ = join_op_(*r_, *x_);
      ++r_;
    };

    //! Performs the restrict-join operation for an arbitrary result (slow).
    void loop() {
      x_inc_.partial_sum();
      uint_vector_iterator it(shape_), end(shape_.size());
      while (it != end) {
        *r_ = join_op_(*r_, *x_);
        ++r_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const uint_vector& shape_;
    T* r_;
    const T* x_;
    table_increment x_inc_;
    JoinOp join_op_;

  }; // class table_restrict_join

} // namespace libgm

#endif
