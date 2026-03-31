#pragma once

#include <boost/container_hash/hash.hpp>

#include <compare>
#include <cstddef>
#include <iosfwd>

namespace libgm {

struct GridArg {
  std::size_t row = 0;
  std::size_t col = 0;

  auto operator<=>(const GridArg&) const = default;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(row, col);
  }
};

inline std::ostream& operator<<(std::ostream& out, const GridArg& arg) {
  return out << '(' << arg.row << ", " << arg.col << ')';
}

inline std::size_t hash_value(const GridArg& arg) noexcept {
  std::size_t seed = 0;
  boost::hash_combine(seed, arg.row);
  boost::hash_combine(seed, arg.col);
  return seed;
}

} // namespace libgm

namespace std {

template <>
struct hash<libgm::GridArg> {
  std::size_t operator()(const libgm::GridArg& arg) const noexcept {
    return libgm::hash_value(arg);
  }
};

} // namespace std
