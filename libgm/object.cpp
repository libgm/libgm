#include "object.hpp"

namespace libgm {

Object::Object(const Object& other)
  : impl_(other.impl_->clone())

Object& Object::operator=(const Object& other) {
  impl_.reset(other.impl_->clone());
  return *this;
}

void Object::save(oarchive& ar) const {
  // TODO: save typeid
  if (impl_) {
    impl_->save(ar);
  }
}

void Object::load(iarchive& ar) {
  // TODO: load typeid and create the Impl object
  if (impl_) {
    impl_->load(ar);
  }
}

std::ostream& operator<<(std::ostream& out, const Object& object) {
  if (impl_) {
    impl_->print(out);
  } else {
    out << "null";
  }
  return out;
}

} // namespace libgm
