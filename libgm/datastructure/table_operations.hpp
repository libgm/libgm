#include <libgm/argument/shape.hpp>
#include <libgm/datastructure/table.hpp>

#include <cereal/cereal.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>

namespace libgm {

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
class TableIncrement {
public:
  /// Creates an increment for the given shape of table A, selecting the given dimensions of table B.
  TableIncrement(const Shape& a_shape, const Dims& b_dims);

  /// Creates an increment omitting the given dimensions of table A, given the shape of table B.
  TableIncrement(const Dims& a_dims, const Shape& b_shape);

  /**
   * Destructor.
   */
  ~TableIncrement();

  /**
   * Returns the increment for the given dimension.
   */
  ptrdiff_t operator[](size_t i) const {
    return inc_[i];
  }

  /**
   * Returns the number of dimensions of the incremnet.
   */
  size_t size() const {
    return size_;
  }

private:
  explicit TableIncrement(size_t arity);

  /// Size of the increment array.
  size_t size_;

  /// Pre-allocated buffer for tables with few dimensions.
  ptrdiff_t buf_[6];

  /// Pointer to the increment array (buffer, or allocated on the heap).
  ptrdiff_t* inc_;

};  // class TableIncrement

/// Prints the increment to an output stream.
std::ostream& operator<<(std::ostream& out, const TableIncrement& inc);

/**
 * A simple class that stores the index of a table.
 */
class TableIndex {
public:
  /// Constructs an index with given number of dimensions.
  explicit TableIndex(size_t arity);

  /// Destructor.
  ~TableIndex();

  size_t operator[](size_t i) const {  return data_[i]; }

  /// Advances the index by 1 and returns the position of the last updated index.
  size_t advance(const Shape& shape);

private:
  /// Size of the index array.
  size_t size_;

  /// Pre-allocated buffer for tables with few dimensions.
  size_t buf_[6];

  /// Pointer to the index array (buffer, or allocated on the heap).
  size_t* data_;

  /// Prints the increment to an output stream.
  friend std::ostream& operator<<(std::ostream& out, const TableIndex& index);

};  // class TableIndex

// Operations
//------------------------------------------------------------------------------

template <typename T, typename UnaryOp>
void transform_in(Table<T>& table, UnaryOp op) {
  std::transform(table.begin(), table.end(), table.begin(), op);
}

template <typename T, typename BinaryOp>
void transform_in(Table<T>& a, const Table<T>& b, BinaryOp op) {
  assert(a.shape() == b.shape());
  std::transform(a.begin(), a.end(), b.begin(), a.begin(), op);
}

template <typename T, typename UnaryOp>
void transform(const Table<T>& table, UnaryOp op, Table<T>& result) {
  result.reset(table.shape());
  std::transform(table.begin(), table.end(), result.begin(), op);
}

template <typename T, typename BinaryOp>
void transform(const Table<T>& a, const Table<T>& b, BinaryOp op, Table<T>& result) {
  assert(a.shape() == b.shape());
  result.reset(a.shape());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), op);
}

template <typename T, typename BinaryOp>
void join_in_front(Table<T>& a, const Table<T>& b, BinaryOp op) {
  assert(a.shape().has_prefix(b.shape()));
  size_t step = b.size();
  for (auto it = a.begin(); it != a.end(); it += step) {
    std::transform(it, it + step, b.begin(), it, op);
  }
}

template <typename T, typename BinaryOp>
void join_in_back(Table<T>& a, const Table<T>& b, BinaryOp op) {
  assert(a.shape().has_suffix(b.shape()));
  auto a_it = a.begin();
  size_t step = a.size() / b.size();
  for (auto b_it = b.begin(); b_it != b.end(); ++b_it) {
    for (size_t i = 0; i < step; ++i) {
      *a_it = op(*a_it, *b_it);
      ++a_it;
    }
  }
}

template <typename T, typename BinaryOp>
void join_in(Table<T>& a, const Table<T>& b, const Dims& dims, BinaryOp op) {
  // Initialize the indexing arrays
  TableIncrement b_inc(a.shape(), dims);
  TableIndex index(a.arity());

  // Join the tables, by iterating over A sequentially and over B repeataedly.
  auto b_it = b.begin();
  for (auto a_it = a.begin(); a_it != a.end(); ++a_it) {
    *a_it = op(*a_it, *b_it);

    // Advance the source pointer, as given by the highest incremented index position.
    size_t i = index.advance(a.shape());
    b_it += b_inc[i];
  }
}

template <typename T, typename BinaryOp>
void join_front(const Table<T>& a, const Table<T>& b, BinaryOp op, Table<T>& result) {
  assert(a.shape().has_prefix(b.shape()));
  result.reset(a.shape());
  auto dest = result.begin();
  size_t step = b.size();
  for (auto it = a.begin(); it != a.end(); it += step) {
    dest = std::transform(it, it + step, b.begin(), dest, op);
  }
}

template <typename T, typename BinaryOp>
void join_back(const Table<T>& a, const Table<T>& b, BinaryOp op, Table<T>& result) {
  assert(a.shape().has_suffix(b.shape()));
  result.reset(a.shape());
  auto a_it = a.begin();
  auto dest = result.begin();
  size_t step = a.size() / b.size();
  for (auto b_it = b.begin(); b_it != b.end(); ++b_it) {
    for (size_t i = 0; i < step; ++i) {
      *dest++ = op(*a_it++, *b_it);
    }
  }
}

template <typename T, typename BinaryOp>
void join(const Table<T>& a, const Table<T>& b, const Dims& i, const Dims& j, BinaryOp op, Table<T>& result) {
  result.reset(libgm::join(a.shape(), b.shape(), i, j));

  // Initialize the indexing arrays
  TableIncrement a_inc(result.shape(), i);
  TableIncrement b_inc(result.shape(), j);
  TableIndex index(result.arity());

  // Join the tables, by iterating over the result sequentially and over a and b repeataedly.
  auto a_it = a.begin();
  auto b_it = b.begin();
  for (auto dest = result.begin(); dest != result.end(); ++dest) {
    *dest = op(*a_it, *b_it);

    // Advance the source pointers, as given by the highest incremented index position.
    size_t i = index.advance(result.shape());
    a_it += a_inc[i];
    b_it += b_inc[i];
  }
}

template <typename T, typename BinaryOp>
void aggregate_front(const Table<T>& a, unsigned n, T init, BinaryOp op, Table<T>& result) {
  result.reset(a.shape().prefix(n));
  result.fill(init);
  size_t step = result.size();
  for (auto it = a.begin(); it != a.end(); it += step) {
    std::transform(result.begin(), result.end(), it, result.begin(), op);
  }
}

template <typename T, typename BinaryOp>
void aggregate_back(const Table<T>& a, unsigned n, T init, BinaryOp op, Table<T>& result) {
  result.reset(a.shape().suffix(n));
  auto dest = result.begin();
  size_t step = a.size() / result.size();
  for (auto it = a.begin(); it != a.end(); it += step) {
    *dest++ = std::accumulate(it, it + step, init, op);
  }
}

template <typename T, typename BinaryOp>
void aggregate(const Table<T>& a, const Dims& retain, T init, BinaryOp op, Table<T>& result) {
  result.reset(a.shape().select(retain));
  result.fill(init);

  // Initialize the indexing arrays
  TableIncrement inc(a.shape(), retain);
  TableIndex index(a.arity());

  // Aggregate the values, by iterating over table A sequentially and result "jumpy"
  auto dest = result.begin();
  for (auto it = a.begin(); it != a.end(); ++it) {
    *dest = op(*dest, *it);

    // Advance the destination based on the highest incremented index position
    size_t i = index.advance(a.shape());
    dest += inc[i];
  }
}

template <typename T>
void restrict_front(const Table<T>& a, const std::vector<size_t>& values, Table<T>& result) {
  result.reset(a.shape().suffix(a.arity() - values.size()));
  auto it = a.begin() + a.shape().linear_front(values);
  size_t step = a.size() / result.size();
  for (auto dest = result.begin(); dest != result.end(); ++dest) {
    *dest = *it;
    it += step;
  }
}

template <typename T>
void restrict_back(const Table<T>& a, const std::vector<size_t>& values, Table<T>& result) {
  result.reset(a.shape().prefix(a.arity() - values.size()));
  auto it = a.begin() + a.shape().linear_back(values);
  std::copy(it, it + result.size(), result.begin());
}

template <typename T>
void restrict(const Table<T>& a, const Dims& dims, const std::vector<size_t>& values, Table<T>& result) {
  assert(dims.count() == values.size());
  result.reset(a.shape().omit(dims));

  // Initialize the indexing arrays
  TableIncrement inc(dims, a.shape());
  TableIndex index(result.arity());

  // Extract the values, starting from the position of the first value.
  auto it = a.begin() + a.shape().linear(dims, values);
  for (auto dest = result.begin(); dest != result.end(); ++dest) {
    *dest = *it;

    // Advance the source pointer, as given by the highest incremented index position.
    size_t i = index.advance(result.shape());
    it += inc[i];
  }
}

template <typename ARCHIVE, typename T>
void save(ARCHIVE& ar, const Table<T>& table) {
  ar(table.shape());
  ar(cereal::make_size_tag(table.size()));
  for (const T& value : table) ar(value);
}

template <typename ARCHIVE, typename T>
void load(ARCHIVE& ar, Table<T>& table) {
  Shape shape;
  cereal::size_type size;
  ar(shape);
  ar(cereal::make_size_tag(size));
  table.reset(std::move(shape));
  assert(table.size() == size);
  for (T& value : table) ar(value);
}

} // namespace libgm
