#pragma once

#include <libgm/argument/domain.hpp>

#include <type_traits>
#include <utility>

namespace libgm {

template <typename T, typename Property = void>
struct Annotated {
  T value;
  Property property_;

  Annotated() = default;

  template <typename U, typename P>
  Annotated(U&& value, P&& property)
    : value(std::forward<U>(value)),
      property_(std::forward<P>(property)) {}

  Property& property() {
    return property_;
  }

  const Property& property() const {
    return property_;
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
