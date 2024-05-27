#include <functional>
#include <cstddef>

#include <libgm/config.hpp>

namespace libgm {

class Shape {
public:
  Shape(const size_t* begin, size_t* end)
    : begin_(begin), size_(end - begin) {}

  Shape(size_t size)
    : begin_(nullptr), size_(size) {}

  bool is_real() const {
    return !begin_;
  }

  bool is_discrete() const {
    return begin_;
  }

  const size_t* begin() const {
    return begin_;
  }

  const size_t* end() const {
    return begin_ + size_;
  }

private:
  const size_t* begin_;
  size_t size_;
;

}; // class Shape

using ShapeMap = std::function<Shape(Arg arg)>;

} // namespace libgm
