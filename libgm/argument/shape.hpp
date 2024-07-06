#pragma once

#include <libgm/argument/argument.hpp>
#include <libgm/argument/dims.hpp>

#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

namespace libgm {

/**
 * A vector of sizes.
 */
class Shape : public std::vector<size_t> {
public:
  // Bring constructors from the base class.
  using std::vector<size_t>::vector;

  /// Returns the prefix of the shape.
  Shape prefix(unsigned n) const;

  /// Returns the suffix of the shape.
  Shape suffix(unsigned n) const;

  /// Returns a subsequence of the shape for the given dimensions.
  Shape subseq(const Dims& dims) const;

  /// Returns the sum of the prefix of the shape.
  size_t prefix_sum(unsigned n) const;

  /// Returns the sum of the suffix of the shape.
  size_t suffix_sum(unsigned n) const;

  /// Returns the sum of all elements in the shape.
  size_t sum() const;

  /**
   * Returns the index spans for the given dimensions.
   *
   * Note that consecutive spans may get concatenated, so the size of the returned
   * vector may not match the number of dimensions in dims.
   */
  Spans spans(const Dims& dims, bool ignore_out_of_range = false) const;
};

// Concatenats two shapes.
Shape operator+(const Shape& a, const Shape& b);

/// Joins two shapes on the given dimensions.
Shape join(const Shape& a, const Shape& b, const Dims& i, const Dims& j);

/// Prints the shape to an output stream.
std::ostream& operator<<(std::ostream& out, const Shape& shape);

/// A mapping from an argument to its size.
using ShapeMap = std::function<size_t(Arg arg)>;

} // namespace libgm
