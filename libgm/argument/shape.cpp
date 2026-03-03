#include "shape.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace libgm {

bool Shape::has_prefix(const Shape& other) const {
  return other.size() <= size() && std::equal(other.begin(), other.end(), begin());
}

bool Shape::has_suffix(const Shape& other) const {
  return other.size() <= size() && std::equal(other.begin(), other.end(), end() - other.size());
}

bool Shape::has_select(const Dims& dims, const Shape& other) const {
  assert(dims.count() == other.size());

  // Is the other shape longer?
  if (other.size() > size()) {
    return false;
  }

  // Does the other shape match at dimensions dims?
  size_t j = 0;
  for (size_t i = 0; i < size(); ++i) {
    if (dims.test(i) && operator[](i) != other[j++]) {
      return false;
    }
  }

  // Have we seen all set dimensions?
  return j == other.size();
}

Shape Shape::prefix(unsigned n) const {
  if (n > size()) {
    throw std::invalid_argument("Shape::prefix: Prefix size exceeds shape size.");
  }
  return {begin(), begin() + n};
}

Shape Shape::suffix(unsigned n) const {
  if (n > size()) {
    throw std::invalid_argument("Shape::suffix: Suffix size exceeds shape size.");
  }
  return {end() - n, end()};
}

Shape Shape::select(const Dims& dims) const {
  // Preallocate the result
  Shape result(dims.count());

  // Copy over the shape for the set bits.
  size_t i, j;
  for (i = 0, j = 0; i < size(); ++i) {
    if (dims.test(i)) {
      result[j++] = operator[](i);
    }
  }

  // Check if we covered all the set bits.
  if (j < result.size()) {
    throw std::invalid_argument("Shape::select: Dims out of range.");
  }

  return result;
}

Shape Shape::omit(const Dims& dims) const {
  // Preallocate the result
  Shape result(size() > dims.count() ? size() - dims.count() : 0);

  // Copy over the shape for the unset bits.
  for (size_t i = 0, j = 0; i < size(); ++i) {
    if (!dims.test(i)) {
      if (j >= result.size()) {
        throw std::invalid_argument("Shape::omit Dims out of range.");
      }
      result[j++] = operator[](i);
    }
  }

  return result;
}

size_t Shape::prefix_sum(unsigned n) const {
  if (n > size()) {
    throw std::invalid_argument("Shape::prefix_sum: Prefix size exceeds shape size.");
  }
  return std::accumulate(begin(), begin() + n, size_t(0));
}

size_t Shape::suffix_sum(unsigned n) const {
  if (n > size()) {
    throw std::invalid_argument("Shape::suffix_sum: Suffix size exceeds shape size.");
  }
  return std::accumulate(end() - n, end(), size_t(0));
}

size_t Shape::sum() const {
  return std::accumulate(begin(), end(), size_t(0));
}

size_t Shape::product() const {
  size_t result = 1;
  for (size_t i = 0; i < size(); ++i) {
    if (std::numeric_limits<size_t>::max() / operator[](i) <= result) {
      throw std::out_of_range("Shape::product likely overflows size_t");
    }
    result *= operator[](i);
  }
  return result;
}

std::vector<size_t> Shape::index(size_t linear) const {
  std::vector<size_t> result(size());
  for (size_t i = 0; i < size(); ++i) {
    result[i] = linear % operator[](i);
    linear /= operator[](i);
  }
  if (linear) {
    throw std::invalid_argument("Shape::index: the given linear index is out of range.");
  }
  return result;
}

size_t Shape::linear(const std::vector<size_t>& index) const {
  if (size() != index.size()) {
    throw std::invalid_argument("Shape::linear: mismatch of the index and shape sizes");
  }

  // Compute the linear index for all dimensions.
  size_t result = 0;
  for (ptrdiff_t i = size() - 1; i >= 0; --i) {
    if (index[i] >= operator[](i)) {
      throw std::invalid_argument("Shape::linear: index " + std::to_string(i) + " out of range");
    }
    result *= operator[](i);
    result += index[i];
  }

  return result;
}

size_t Shape::linear_front(const std::vector<size_t>& index) const {
  if (index.size() > size()) {
    throw std::invalid_argument("Shape::linear_front: index size exceeds shape size");
  }

  // Compute the linear index for the index.size() front dimensions.
  // The remaining dimensions are implicitly zero.
  size_t result = 0;
  for (ptrdiff_t i = index.size() - 1; i >= 0; --i) {
    if (index[i] >= operator[](i)) {
      throw std::invalid_argument(
        "Shape::linear_front: index " + std::to_string(i) + " out of range");
    }
    result *= operator[](i);
    result += index[i];
  }

  return result;
}

size_t Shape::linear_back(const std::vector<size_t>& index) const {
  if (index.size() > size()) {
    throw std::invalid_argument("Shape::linear_back: index size exceeds shape size");
  }
  size_t offset = size() - index.size();

  // Compute the linear index for the index.size() trailing dimensions.
  size_t result = 0;
  for (ptrdiff_t i = index.size() - 1; i >= 0; --i) {
    if (index[i] >= operator[](offset + i)) {
      throw std::invalid_argument(
        "Shape::linear_back: index " + std::to_string(i) + " out of range");
    }
    result *= operator[](offset + i);
    result += index[i];
  }

  // Shift by the remaining front dimensions (these are implicitly zero).
  for (size_t i = 0; i < offset; ++i) {
    result *= operator[](i);
  }

  return result;
}

size_t Shape::linear(const Dims& dims, const std::vector<size_t>& index) const {
  // Dimensions check - overall count
  if (dims.count() != index.size()) {
    throw std::invalid_argument("Shape::linear: index size does not match dimension count");
  }

  // Dimensions check - count in the leading size() dimensions
  size_t cnt = 0;
  for (size_t i = 0; i < size(); ++i) {
    cnt += dims.test(i);
  }
  if (cnt != index.size()) {
    throw std::invalid_argument("Shape::linear: index size does not match leading dimensions");
  }

  // Compute the linear index
  size_t result = 0;
  for (ptrdiff_t i = size() - 1, j = index.size() - 1; i >= 0; --i) {
    result *= operator[](i);
    if (dims.test(i)) {
      if (index[j] >= operator[](i)) {
        throw std::invalid_argument("Shape::linaer: index " + std::to_string(j) + " out of range");
      }
      result += index[j--];
    }
  }

  return result;
}

Spans Shape::spans(const Dims& dims, bool ignore_out_of_range) const {
  // Reserve enough spans in the result (worst-case, one per dimension).
  Spans result;
  result.reserve(std::min(size(), dims.count()));

  // Walk over the dimensions, accumulating the sizes.
  Span span;
  size_t count = 0;
  for (size_t i = 0; i < size(); ++i) {
    if (dims.test(i)) {
      ++count;
      if (span.length == 0) span.start = i;
      span.length += operator[](i);
    } else if (span.length > 0) {
      result.push_back(span);
      span.length = 0;
    }
  }

  // Check if we covered all the set bits.
  if (!ignore_out_of_range && count < dims.count()) {
    throw std::invalid_argument("Shape::spans: Dims out of range.");
  }

  // Push the final one
  if (span.length > 0) {
    result.push_back(span);
  }

  return result;
}

Shape operator+(const Shape& a, const Shape& b) {
  Shape result(a.size() + b.size());
  std::copy(a.begin(), a.end(), result.begin());
  std::copy(b.begin(), b.end(), result.begin() + a.size());
  return result;
}

Shape join(const Shape& a, const Shape& b, const Dims& i, const Dims& j) {
  // Check that the provided dims are consistent.
  if (a.size() != i.size()) {
    throw std::invalid_argument("Shape join: Inconsistent number of left dimensions.");
  }
  if (b.size() != j.size()) {
    throw std::invalid_argument("Shape join: Inconsistent number of right dimensions.");
  }

  // Initialize the result.
  Shape result((i | j).count());

  // Copy over the shapes, checking if the left and right ones are consistent.
  auto a_it = a.begin();
  auto b_it = b.begin();
  for (size_t k = 0; k < result.size(); ++k) {
    if (i.test(k)) {
      result[k] = *a_it++;
      if (j.test(k) && result[k] != *b_it++) {
        throw std::invalid_argument("Shape join: Inconsistent left and right dimensions.");
      }
    } else if (j.test(k)) {
      result[k] = *b_it++;
    }
  }

  // Check if we covered all left and right dimensions.
  if (a_it != a.end()) {
    throw std::invalid_argument("Shape join: Left dimensions out of range.");
  }
  if (b_it != b.end()) {
    throw std::invalid_argument("Shape join: Right dimensions out of range.");
  }

  // TODO: prove that all shapes are populated and valid.
  return result;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << "Shape([";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out << ", ";
    out << shape[i];
  }
  out << "])";
  return out;
}

} // namespace libgm
