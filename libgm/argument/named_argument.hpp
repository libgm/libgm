#pragma once

#include <libgm/argument/concepts/argument.hpp>

#include <boost/container_hash/hash.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <string_view>

namespace libgm {

template <std::size_t Size>
struct NamedArg {
  static_assert(Size > 0, "NamedArg buffer must have positive size");

  std::array<char, Size> buffer{};

  NamedArg() = default;

  NamedArg(std::string_view name) {
    assign(name);
  }

  NamedArg(const std::string& name)
    : NamedArg(std::string_view(name)) {}

  NamedArg(const char* name)
    : NamedArg(name ? std::string_view(name) : std::string_view()) {}

  void assign(std::string_view name) {
    if (name.size() > Size - 1) {
      throw std::length_error("NamedArg: name exceeds fixed buffer");
    }
    buffer.fill('\0');
    std::memcpy(buffer.data(), name.data(), name.size());
  }

  const char* c_str() const {
    return buffer.data();
  }

  std::string_view view() const {
    return std::string_view(buffer.data());
  }

  auto operator<=>(const NamedArg&) const = default;

  template <typename Archive>
  void save(Archive& ar) const {
    ar(std::string_view(buffer.data()));
  }

  template <typename Archive>
  void load(Archive& ar) {
    std::string value;
    ar(value);
    assign(value);
  }
};

template <std::size_t Size>
std::ostream& operator<<(std::ostream& out, const NamedArg<Size>& arg) {
  return out << arg.c_str();
}

template <std::size_t Size>
std::size_t hash_value(const NamedArg<Size>& arg) noexcept {
  return boost::hash_range(arg.buffer.begin(), arg.buffer.end());
}

} // namespace libgm

namespace std {

template <std::size_t Size>
struct hash<libgm::NamedArg<Size>> {
  std::size_t operator()(const libgm::NamedArg<Size>& arg) const noexcept {
    return libgm::hash_value(arg);
  }
};

} // namespace std
