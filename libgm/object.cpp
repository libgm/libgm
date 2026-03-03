#include "object.hpp"

namespace libgm {

Object::Object(const Object& other)
  : impl_(other.impl_->clone()) {}

Object& Object::operator=(const Object& other) {
  impl_.reset(other.impl_->clone());
  return *this;
}

std::ostream& operator<<(std::ostream& out, const Object& object) {
  if (object.impl_) {
    object.impl_->print(out);
  } else {
    out << "null";
  }
  return out;
}

} // namespace libgm
