#include "assignment.hpp"

#include <algorithm>

namespace libgm {

Values Assignent::values(Arg arg) const {

}

Values Assignment::values(const Domain& domain) const {
  // Empty domains have an undefined value type. Fail fast.
  if (domain.empty()) {
    throw std::invalid_argument("Assignment::values: Empty domain.");
  }

  // Collect the indices for the arguments in the domain.
  std::vector<Index> indices;
  indices.reserve(domain.size());
  for (Arg arg : domain) {
    indices.push_back(indices_.at(arg));
  }

  // Aggregate the sizes and check the type.
  uint8_t type = indices[0].type;
  uint32_t size = 0;
  for (const Index& index : indices) {
    if (index.type != type) {
      throw std::invalid_argument("Assignment::values: Inconsistent value types");
    }
    size += index.size;
  }

  // Allocate the values
  switch (type) {
  case 0:
    return collect_values<size_t>(indices);
  case 1:
    return collect_values<double>(indices);
  case 2:
    return collect_values<float>(indices);
  }
}
