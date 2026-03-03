#include "table_operations.hpp"

namespace libgm {

TableIncrement::TableIncrement(const Shape& a_shape, const Dims& b_dims)
  : TableIncrement(a_shape.size()) {
  // The increment is 1 for all present dimensions, and -multiplier + 1 for the others.
  ptrdiff_t multiplier = 1;
  for (size_t i = 0; i < a_shape.size(); ++i) {
    if (b_dims.test(i)) {
      inc_[i] = 1;
      multiplier *= a_shape[i];
    } else {
      inc_[i] = -multiplier + 1;
    }
  }
}

TableIncrement::TableIncrement(const Dims& a_dims, const Shape& b_shape)
  : TableIncrement(b_shape.size() - a_dims.count()) {
  std::fill_n(inc_, size_, ptrdiff_t(0));
  ptrdiff_t multiplier = 1;
  for (size_t i = 0, j = 0; i < b_shape.size(); ++i) {
    if (a_dims.test(i)) {
      // rewind at the next position if we are transitioning from unrestricted dims
      if (i != 0 && !a_dims.test(i - 1)) inc_[j] = multiplier;
    } else {
      // advance at this position if we are transitioning from restricted dims
      if (i == 0 || a_dims.test(i - 1)) inc_[j] += multiplier;
      ++j;
    }
    multiplier *= b_shape[i];
  }

  // Add up the contributions up to the last incremented dimension.
  std::partial_sum(inc_, inc_ + size_, inc_);
}

TableIncrement::TableIncrement(size_t arity)
  : size_(arity) {
  if (arity > 6) {
    inc_ = new ptrdiff_t[arity];
  } else {
    inc_ = buf_;
  }
}

TableIncrement::~TableIncrement() {
  if (inc_ != buf_) {
    delete[] inc_;
  }
}

std::ostream& operator<<(std::ostream& out, const TableIncrement& inc) {
  out << "[";
  for (size_t i = 0; i < inc.size(); ++i) {
    if (i > 0) out << ", ";
    out << inc[i];
  }
  out << "]";
  return out;
}

TableIndex::TableIndex(size_t arity)
  : size_(arity) {
  if (arity > 6) {
    data_ = new size_t[arity];
  } else {
    data_ = buf_;
  }
  std::fill_n(data_, arity, size_t(0));
}

TableIndex::~TableIndex() {
  if (data_ != buf_) {
    delete[] data_;
  }
}

size_t TableIndex::advance(const Shape& shape) {
  assert(size_ == shape.size());
  for (size_t i = 0; i < size_; ++i) {
    if (++data_[i] == shape[i]) {
      data_[i] = 0;
    } else {
      return i;
    }
  }
  return size_;
}

std::ostream& operator<<(std::ostream& out, const TableIndex& index) {
  out << "[";
  for (size_t i = 0; i < index.size_; ++i) {
    if (i > 0) out << ", ";
    out << index.data_[i];
  }
  out << "]";
  return out;
}

} // namespace libgm
