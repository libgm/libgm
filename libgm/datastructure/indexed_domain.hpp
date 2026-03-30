#pragma once

#include <libgm/argument/domain.hpp>
#include <libgm/datastructure/intrusive_list.hpp>

#include <cereal/cereal.hpp>

namespace libgm {

template <typename Item>
struct IndexedDomain {
  Item* owner = nullptr;
  Domain args;
  typename IntrusiveList<IndexedDomain<Item>>::HookArray hooks;

  IndexedDomain() = default;

  explicit IndexedDomain(Domain args)
    : args(std::move(args)),
      hooks(this->args.size()) {}

  IndexedDomain(const IndexedDomain& other)
    : owner(other.owner),
      args(other.args),
      hooks(args.size()) {}

  IndexedDomain& operator=(const IndexedDomain& other) {
    if (this != &other) {
      owner = other.owner;
      args = other.args;
      hooks.reset(args.size());
    }
    return *this;
  }

  IndexedDomain(IndexedDomain&&) noexcept = default;
  IndexedDomain& operator=(IndexedDomain&&) noexcept = default;

  Item* item() const {
    return owner;
  }

  const Domain& domain() const {
    return args;
  }

  void reset(Domain new_args) {
    args = std::move(new_args);
    hooks.reset(args.size());
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("domain", args));
    if constexpr (Archive::is_loading::value) {
      hooks.reset(args.size());
    }
  }
};

} // namespace libgm
