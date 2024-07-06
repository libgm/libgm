#include "shape.hpp"

#include <algorithm>
#include <numeric>

namespace libgm {

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

Shape Shape::subseq(const Dims& dims) const {
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
    throw std::invalid_argument("Shape::subseq: Dims out of range.").
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

std::vector<Span> Shape::spans(const Dims& dims, bool ignore_out_of_range) const {
  // Reserve enough spans in the result (worst-case, one per dimension).
  std::vector<Span> result;
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
    throw std::invalid_argument("Shape join: Left dimensions out of range.")
  }
  if (b_it != b.end()) {
    throw sdt::invalid_argument("Shape join: Right dimensions out of range.");
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

}
