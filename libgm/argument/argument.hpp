#pragma once

#include <ankerl/unordered_dense.h>

#include <boost/operators.hpp>

#include <cereal/access.hpp>

#include <functional>
#include <iostream>
#include <memory>

namespace libgm {

struct Arg : boost::less_than_comparable<Arg>, boost::equality_comparable<Arg>{
  char label : 8;
  uint32_t id : 24;
  uint32_t index : 32;

  // Constructs a null argument.
  Arg()
    : label(0), id(0), index(0) {}

  // Constructs an argument with given parameters.
  Arg(char label, uint32_t id, uint32_t index)
    : label(label), id(id), index(index) {}

  /**
   * Sets the global ordering of arguments by their label.
   */
  static void set_ordering(const std::string& ordering);

  /**
   * Resets the global ordering of labels to the alphabetical ordering.
   */
  static void reset_ordering();

private:
  friend class cereal::access;

  /// Serialize the argument.
  template <typename ARCHIVE>
  void save(ARCHIVE& archive) const {
    archive(label, id, index);
  }

  /// Deserialize the argument.
  template <typename ARCHIVE>
  void load(ARCHIVE& archive) {
    char label;
    uint32_t id, index;
    archive(label, id, index);
    this->label = label;
    this->id = id;
    this->index = index;
  }

  /// The global priority
  static std::unique_ptr<int[]> priority;

  friend bool operator<(const Arg, const Arg);
};

std::ostream& operator<<(std::ostream& out, Arg arg);

inline bool operator==(Arg a, Arg b) {
  return reinterpret_cast<uint64_t&>(a) == reinterpret_cast<uint64_t&>(b);
}

inline bool operator<(const Arg a, const Arg b) {
  if (a.label == b.label) {
    // priorities are always unique, delegate to the remaining fields
    return std::make_pair(a.id, a.index) < std::make_pair(b.id, b.index);
  } else {
    // use priorities if present, otherwise compare alphabetically
    return Arg::priority ? Arg::priority[a.label] < Arg::priority[b.label] : a.label < b.label;
  }
}

/**
 * An unordered set of argumetns.
 */
using ArgSet = ankerl::unordered_dense::set<Arg>;

}  // namespace libgm

namespace std {

template <>
struct hash<libgm::Arg> {
  size_t operator()(libgm::Arg arg) const noexcept {
    return hash<uint64_t>()(reinterpret_cast<const uint64_t&>(arg));
  }
};

} // namespace std
