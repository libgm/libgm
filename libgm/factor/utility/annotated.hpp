#pragma once

#include <libgm/argument/domain.hpp>

#include <type_traits>
#include <utility>

namespace libgm {

template <typename T, typename Property = void>
struct Annotated : Property {
  T value;

  Annotated() = default;

  template <typename U, typename P>
  Annotated(U&& value, P&& property)
    : Property(std::forward<P>(property)),
      value(std::forward<U>(value)) {}

  Property& property() {
    return *this;
  }

  const Property& property() const {
    return *this;
  }
};

template <typename T>
struct Annotated<T, void> {
  T value;

  Annotated() = default;

  template <typename U>
  explicit Annotated(U&& value)
    : value(std::forward<U>(value)) {}

  void property() {}

  void property() const {}
};

template <typename Factor>
Annotated<std::decay_t<Factor>, Domain> annotate(Factor&& factor, Domain domain) {
  return {std::forward<Factor>(factor), std::move(domain)};
}

}
