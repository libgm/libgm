#include <functional>
#include <cstddef>

#include <libgm/config.hpp>

namespace libgm {

using ShapeMap = std::function<size_t(Arg arg)>;

} // namespace libgm
