#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <concepts>
#include <iosfwd>

namespace libgm {

namespace detail {

template <typename T>
concept CerealOutputSerializable =
  requires(std::ostream& os, const T& value) {
    cereal::BinaryOutputArchive(os)(value);
  };

template <typename T>
concept CerealInputSerializable =
  requires(std::istream& is, T& value) {
    cereal::BinaryInputArchive(is)(value);
  };

template <typename T>
concept CerealSerializable =
  CerealOutputSerializable<T> && CerealInputSerializable<T>;

} // namespace detail

template <typename T>
concept Argument =
  std::copyable<T> &&
  std::equality_comparable<T> &&
  std::totally_ordered<T> &&
  detail::CerealSerializable<T> &&
  requires(std::ostream& out, const T& value) {
    { out << value } -> std::same_as<std::ostream&>;
  };

} // namespace libgm
