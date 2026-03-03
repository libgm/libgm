#pragma once

#include <libgm/object.hpp>
#include <libgm/math/eigen/dense.hpp>

#include <initializer_list>

namespace libgm {

template <typename T>
class RealValues : public Object {
public:
  struct Impl;

  RealValues(std::initializer_list<T> list);
  RealValues(Vector<T> values);

  RealValues& operator=(Vector<T> other);

  size_t size() const;
  T operator[](size_t pos) const;
  T& operator[](size_t pos);
  T* resize(size_t size);
  const T* data() const;
  const Vector<T>& vec() const;

private:
  Impl& impl();
  const Impl& impl() const;
};

} // namespace libgm
