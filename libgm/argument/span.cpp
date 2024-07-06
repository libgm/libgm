#include "span.hpp"

#include <numeric>

namespace libgm {

std::ostream& operator<<(std::ostream& out, const Span& span) {
  out << '(' << span.start << ", " << span.length << ')';
  return out;
}

std::ostream& operator<<(std::ostream& out, const Spans& spans) {
  out << "Spans([";
  for (size_t i = 0; i < spans.size(); ++i) {
    if (i > 0) out << ", ";
    out << spans[i];
  }
  out << "])";
  return out;
}

size_t Spans::sum() const {
  return std::accumulate(begin(), end(), size_t(0), [](size_t acc, const Span& span) {
    return acc + span.length;
  });
}

} // namespace libgm
