#pragma once

#include <libgm/object.hpp>

#include <initializer_list>
#include <vector>

namespace libgm {

class DiscreteValues : public Object {
public:
  struct Impl;

  DiscreteValues(std::initializer_list<size_t> list);
  DiscreteValues(std::vector<size_t> values);

  DiscreteValues& operator=(std::vector<size_t> other);

  size_t size() const;
  size_t operator()() const;
  size_t operator[](size_t pos) const;
  size_t& operator[](size_t pos);
  size_t* resize(size_t size);
  const size_t* data() const;
  const std::vector<size_t>& vec() const;

private:
  Impl& impl();
  const Impl& impl() const;
};

} // namespace libgm
