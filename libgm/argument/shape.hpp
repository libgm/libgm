#pragma once

#include <libgm/argument/concepts/argument.hpp>
#include <libgm/argument/dims.hpp>
#include <libgm/argument/span.hpp>

#include <cereal/types/vector.hpp>

#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

namespace libgm {

/**
 * The shape of a tensor or partitioning of a vector.
 */
class Shape : public std::vector<size_t> {
public:
  // Bring constructors from the base class.
  using std::vector<size_t>::vector;

  /// Returns true if this shape has another shape as a prefix.
  bool has_prefix(const Shape& other) const;

  /// Returns true if this shape has another shape as a suffix.
  bool has_suffix(const Shape& other) const;

  /// Returns true if this shape matches another shape at the given dimensions of this shape.
  bool has_select(const Dims& dims, const Shape& other) const;

  /// Returns the prefix of the shape for the given number of dimensions.
  Shape prefix(unsigned n) const;

  /// Returns the suffix of the shape for the given number of dimensions.
  Shape suffix(unsigned n) const;

  /// Returns a subsequence of the shape for the given dimensions.
  Shape select(const Dims& dims) const;

  /// Returns a subsequence of the shape, omitting the given dimensions.
  Shape omit(const Dims& dims) const;

  /// Returns the sum of the prefix of the shape.
  size_t prefix_sum(unsigned n) const;

  /// Returns the sum of the suffix of the shape.
  size_t suffix_sum(unsigned n) const;

  /// Returns the number of prefix elements, whose sum equals the given length.
  /// This can be thought of as the inverse of prefix_sum.
  /// \throw std::invalid_argument if there is no exact match
  unsigned prefix_size(size_t length) const;

  /// Returns the number of suffix elements, whose sum equals the given length.
  /// This can be thought of as the inverse of suffix_sum.
  /// \throw std::invalid_argument if there is no exact match
  unsigned suffix_size(size_t length) const;

  /// Returns the sum of all elements in the shape.
  size_t sum() const;

  /// Returns the product of all elements in the shape, checking for overflow.
  size_t product() const;

  /// Returns the multiplier for the given position. This is the product of elements up to that position.
  size_t multiplier(unsigned n) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table/tensor),
   * converts the given linear index to an index in that array.
   */
  std::vector<size_t> index(size_t linear) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table/tensor),
   * converts the given linear index for the given number of front dimensions to an index in that array.
   */
  std::vector<size_t> index_front(size_t linear, unsigned n) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table/tensor),
   * converts the given index to a linear index in that array.
   */
  size_t linear(const std::vector<size_t>& index) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table/tensor),
   * converts the given index to the front dimensions to a linear index in that array.
   */
  size_t linear_front(const std::vector<size_t>& index) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table / tensor),
   * converts the given index to the back dimensions to a linear index in that array.
   */
  size_t linear_back(const std::vector<size_t>& index) const;

  /**
   * Assuming that this shape represents the shape of a multidimensional array (table / tensor),
   * converts the given index to the selected dimensions to a linear index in that array.
   */
  size_t linear(const Dims& dims, const std::vector<size_t>& index) const;

  /**
   * Returns the index spans for the given dimensions.
   *
   * Note that consecutive spans may get concatenated, so the size of the returned
   * vector may not match the number of dimensions in dims.
   */
  Spans spans(const Dims& dims, bool ignore_out_of_range = false) const;
};

/// Concatenates two shapes.
Shape operator+(const Shape& a, const Shape& b);

/// Joins two shapes on the given dimensions.
Shape join(const Shape& a, const Shape& b, const Dims& i, const Dims& j);

/// Prints the shape to an output stream.
std::ostream& operator<<(std::ostream& out, const Shape& shape);

/// A mapping from an argument to its size.
template <Argument Arg>
using ShapeMap = std::function<size_t(Arg arg)>;

} // namespace libgm
