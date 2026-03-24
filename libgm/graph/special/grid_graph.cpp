#include <libgm/argument/named_argument.hpp>
#include <libgm/graph/special/grid_graph.hpp>

#include <string>

namespace libgm {

Arg make_argument(size_t row, size_t col) {
  return NamedFactory::default_factory().make(
      "(" + std::to_string(row) + ", " + std::to_string(col) + ")");
}

} // namespace libgm
