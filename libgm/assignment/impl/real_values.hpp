#include "../real_values.hpp"

namespace libgm {

template <typename T>
struct RealValues<T>::Impl : Object::Impl {
  Vector<T> values;

  Impl() = default;

  Impl(Vector<T> values)
    : values(std::move(values)) {}

  Impl(std::initializer_list<T> list)
    : values(list.size()) {
    std::copy(list.begin(), list.end(), values.data());
  }

  template <typename ARCHIVE>
  void serialize(ARCHIVE& ar) {
    ar(values);
  }

  // Object operations
  //--------------------------------------------------------------------------

  Impl* clone() const override {
    return new Impl(*this);
  }

  void print(std::ostream& out) const override {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) out << ", ";
      out << values[i];
    }
    out << ']';
  }
};

template <typename T>
RealValues<T>::RealValues(std::initializer_list<T> list)
  : Object(std::make_unique<Impl>(list)) {}

template <typename T>
RealValues<T>::RealValues(Vector<T> values)
  : Object(std::make_unique<Impl>(std::move(values))) {}

template <typename T>
RealValues<T>& RealValues<T>::operator=(Vector<T> other) {
  if (impl_) {
    impl().values = std::move(other);
  } else {
    impl_.reset(new Impl(std::move(other)));
  }
  return *this;
}

template <typename T>
size_t RealValues<T>::size() const {
  return impl().values.size();
}

template <typename T>
T RealValues<T>::operator[](size_t pos) const {
  return impl().values[pos];
}

template <typename T>
T& RealValues<T>::operator[](size_t pos) {
  return impl().values[pos];
}

template <typename T>
const T* RealValues<T>::data() const {
  return impl().values.data();
}

template <typename T>
const Vector<T>& RealValues<T>::vec() const {
  return impl().values;
}

template <typename T>
typename RealValues<T>::Impl& RealValues<T>::impl() {
  return static_cast<Impl&>(*impl_);
}

template <typename T>
const typename RealValues<T>::Impl& RealValues<T>::impl() const {
  return static_cast<Impl&>(*impl_);
}

} // namespace libgm
