#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

namespace libgm {

/**
 * A contiguous sequence of dimensions.
 */
struct Span {
  size_t start = 0;
  size_t length = 0;
};

/// Prints the span to an output stream.
std::ostream& operator<<(std::ostream& out, const Span& span);

/**
 * A list of spans.
 */
struct Spans : std::vector<Span> {
  using std::vector<Span>::vector;

  /// Returns the total length of the spans.
  size_t sum() const;
};

/// Prints the spans to an
std::ostream& operator<<(std::ostream& out, const Spans& spans);

} // namespace libgm
