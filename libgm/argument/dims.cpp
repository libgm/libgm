#include <libgm/argument/dims.hpp>

namespace libgm {

Dims make_dims(std::initializer_list<size_t> idx) {
  Dims dims;
  for (size_t i : idx) {
    dims.set(i);
  }
  return dims;
}

Dims make_dims_range(size_t begin, size_t end) {
  Dims dims;
  for (size_t i = begin; i < end; ++i) {
    dims.set(i);
  }
  return dims;
}

} // namespace libgm
