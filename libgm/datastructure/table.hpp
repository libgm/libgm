#ifndef LIBGM_TABLE_HPP
#define LIBGM_TABLE_HPP

#include <libgm/datastructure/finite_index.hpp>
#include <libgm/datastructure/finite_index_iterator.hpp>
#include <libgm/functional/operators.hpp>
#include <libgm/range/iterator_range.hpp>
#include <libgm/serialization/vector.hpp>

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
   * A class that can perform conversions between finite_index and linear
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
    explicit table_offset(const finite_index& shape) {
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
    void reset(const finite_index& shape) {
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
    std::size_t linear(const finite_index& index) const {
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
    std::size_t linear(const finite_index& index, std::size_t pos) const {
      assert(index.size() + pos <= multiplier_.size());
      std::size_t result = 0;
      for (std::size_t i = 0; i < index.size(); ++i) {
        result += multiplier_[i + pos] * index[i];
      }
      return result;
    }

    /**
     * Returns the finite index for the given linear index.
     */
    finite_index finite(std::size_t offset) const {
      finite_index ind(multiplier_.size());
      // must use int here to avoid wrap-around
      for(int i = multiplier_.size() - 1; i >= 0; --i) {
        ind[i] = offset / multiplier_[i];
        offset = offset % multiplier_[i];
      }
      return ind;
    }

    /**
     * Returns the finite index corresponding to the linear index
     * for the first n dimensions. The specified linear index must
     * not exceed the total number of elements in the first n
     * dimensions.
     */
    finite_index finite(std::size_t offset, std::size_t n) const {
      finite_index ind(n);
      for (int i = n - 1; i >= 0; --i) {
        ind[i] = offset / multiplier_[i];
        offset = offset % multiplier_[i];
      }
      return ind;
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
    table_increment(const finite_index& a_shape,
                    const table_offset& b_offset,
                    const finite_index& b_map)
      : inc_(a_shape.size() + 1, 0) {
      assert(b_offset.size() == b_map.size());
      for (std::size_t i = 0; i < b_map.size(); ++i) {
        if (b_map[i] != std::numeric_limits<std::size_t>::max()) {
          inc_[b_map[i]] = b_offset.multiplier(i);
        }
      }
      for (std::size_t i = a_shape.size(); i > 0; --i) {
        inc_[i] -= inc_[i-1] * a_shape[i-1];
      }
    }

    /**
     * Updates the increment to contain a partial (cumulative) sum
     * of its present values.
     */
    void partial_sum() {
      std::partial_sum(inc_.begin(), inc_.end(), inc_.begin());
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
    //==========================================================================

    //! The type of values stored in this object
    typedef T value_type;

    //! An iterator over the elements of this table in a linear fashion
    typedef T* iterator;

    //! A const iterator over the elements of this table in a linear fashion
    typedef const T* const_iterator;

    //! The type used to represent the indices and shape
    typedef finite_index index_type;

    //! The iterator over all indices in this table
    typedef finite_index_iterator index_iterator;

    // Constructors and initialization
    //==========================================================================

    /**
     * Default constructor. Creates an empty table with no elements.
     */
    table() { }

    /**
     * Constructs a table with the given shape. This constructor does not
     * initialize the table elements. If needed, the table contents can be
     * initialized with other constructors or with the the fill() function.
     */
    explicit table(const finite_index& shape) {
      reset(shape);
    }

    /**
     * Constructs a table with the given shape and initializes its elements
     * to the given value.
     */
    table(const finite_index& shape, const T& init) {
      reset(shape);
      fill(init);
    }

    /**
     * Constructs a table with the given shape and initializes its elements
     * to the given list.
     */
    table(const finite_index& shape, std::initializer_list<T> values) {
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
      finite_index shape;
      ar >> shape;
      reset(shape);
      ar.deserialize_range<T>(data());
    }

    /**
     * Reset the table to the given shape. Allocates the memory, but does
     * not initialize the elements.
     */
    void reset(const finite_index& shape) {
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

    // Accessors
    //==========================================================================

    //! Returns the number of dimensions of this table.
    std::size_t arity() const {
      return shape_.size();
    }

    //! Returns the dimensions of this table.
    const finite_index& shape() const {
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
    finite_index index(const T* it) const {
      assert(it >= begin() && it < end());
      return offset_.finite(it - begin());
    }

    //! Returns an iterator range over indices into this table.
    iterator_range<index_iterator> indices() const {
      typedef iterator_range<index_iterator> range_type;
      if (empty()) {
        return range_type(index_iterator(), index_iterator());
      } else {
        return range_type(index_iterator(&shape_), index_iterator(arity()));
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
    T& operator()(const finite_index& index) {
      return data_[offset_.linear(index)];
    }

    //! Returns the element with the specified finite index
    const T& operator()(const finite_index& index) const {
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
    //==========================================================================
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
    R transform_accumulate(R init,  TransOp trans_op, AccuOp accu_op) const {
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
    finite_index
    sample(TransOp trans, Generator& rng, const finite_index& tail) const {
      assert(tail.size() < arity());
      std::size_t nhead = arity() - tail.size();
      std::size_t nelem = offset().multiplier(nhead);
      const T* elem = data() + offset().linear(tail, nhead);
      T p = std::uniform_real_distribution<T>()(rng);
      for (std::size_t i = 0; i < nelem; ++i) {
        T prob = trans(elem[i]);
        if (p <= prob) {
          return offset().finite(i, nhead);
        } else {
          p -= prob;
        }
      }
      throw std::invalid_argument("The total probability is less than 1");
    }

  private:
    // Private members
    //==========================================================================

    //! The dimensions of this table.
    finite_index shape_;

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
    for (const finite_index& index :  table.indices()) {
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

  /*
      // ConditionalParameter functions
      //====================================================================
      param_type condition(const finite_index& index) const {
        assert(index.size() <= this->arity());
        std::size_t n = this->arity() - index.size();
        finite_index shape(this->shape().begin(), this->shape().begin() + n);
        param_type result(shape);
        result.restrict(*this, this->offset().linear(index, n));
        return result;
      }
  */

  // Table operations
  //==========================================================================

  template <int N>
  using int_ = std::integral_constant<int, N>;

  /**
   * A utility function that invokes the loop template for the table operation
   * class and the given table arity. The goal is to inline the loops if arity
   * is sufficiently small. Specifically, this function invokes the member
   * function template loop<D> with D=arity if arity is small; otherwise,
   * it invokes the non-template loop() member.
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
    default: table_op.loop(); // reached the precompile limit
    }
  }


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
               const finite_index& x_map,
               const finite_index& y_map,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_++ = op_(*x_, *y_);
        ++it;
        x_ += x_inc_[it.digit()];
        y_ += y_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
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
                       const finite_index& x_map,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_ = op_(*r_, *x_);
        ++r_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
    T* r_;
    const T* x_;
    table_increment x_inc_;
    Op op_;

  }; // class table_join_inplace


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
                    const finite_index& result_map,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_ = op_(*r_, *x_++);
        ++it;
        r_ += r_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
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
                         const finite_index& result_map,
                         const finite_index& x_map,
                         const finite_index& y_map,
                         const finite_index& z_shape,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_ = agg_op_(*r_, join_op_(*x_, *y_));
        ++it;
        r_ += r_inc_[it.digit()];
        x_ += x_inc_[it.digit()];
        y_ += y_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
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
   * std::numeric_limits<std::size_t>::max(), then dimension i of x is
   * restricted; otherwise, dimension i of x corresponds to dimension
   * x_map[i] of the result. The argument first specifies the linear
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
                   const finite_index& x_map,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_++ = *x_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
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
   * std::numeric_limits<std::size_t>::max(), then dimension i of x is
   * restricted; otherwise, dimension i of x corresponds to dimension
   * x_map[i] of the result. The argument first specifies the linear
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
                        const finite_index& x_map,
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
      finite_index_iterator it(&shape_), end(shape_.size());
      while (it != end) {
        *r_ = join_op_(*r_, *x_);
        ++r_;
        ++it;
        x_ += x_inc_[it.digit()];
      }
    }

  private:
    const finite_index& shape_;
    T* r_;
    const T* x_;
    table_increment x_inc_;
    JoinOp join_op_;

  }; // class table_restrict_join

} // namespace libgm

#endif
