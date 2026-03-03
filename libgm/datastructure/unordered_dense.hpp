#pragma once

#include <ankerl/unordered_dense.h>

#include <cassert>

namespace ankerl::unordered_dense {

template <typename ARCHIVE,
          typename KEY,
          typename T,
          typename HASH,
          typename EQUAL,
          typename ALLOC,
          typename BUCKET>
void save(ARCHIVE& ar, const map<KEY, T, HASH, EQUAL, ALLOC, BUCKET>& map) {
  ar(cereal::make_size_tag(map.size()));
  for (const auto& [key, value] : map) {
    ar(cereal::make_map_item(key, value));
  }
}

template <typename ARCHIVE,
          typename KEY,
          typename T,
          typename HASH,
          typename EQUAL,
          typename ALLOC,
          typename BUCKET>
void save(ARCHIVE& ar, const map<KEY, T*, HASH, EQUAL, ALLOC, BUCKET>& map) {
  ar(cereal::make_size_tag(map.size()));
  for (const auto& [key, ptr] : map) {
    ar(cereal::make_map_item(key, *ptr));
  }
}

template <typename ARCHIVE,
          typename KEY,
          typename T,
          typename HASH,
          typename EQUAL,
          typename ALLOC,
          typename BUCKET>
void load(ARCHIVE& ar, map<KEY, T, HASH, EQUAL, ALLOC, BUCKET>& map) {
  cereal::size_type size;
  ar(cereal::make_size_tag(size));
  map.clear();
  for (size_t i = 0; i < size; ++i) {
    KEY key;
    T value;
    ar(cereal::make_map_item(key, value));
    map.emplace(std::move(key), std::move(value));
  }
}

template <typename ARCHIVE,
          typename KEY,
          typename T,
          typename HASH,
          typename EQUAL,
          typename ALLOC,
          typename BUCKET>
void load(ARCHIVE& ar, map<KEY, T*, HASH, EQUAL, ALLOC, BUCKET>& map) {
  cereal::size_type size;
  ar(cereal::make_size_tag(size));
  for (size_t i = 0; i < size; ++i) {
    KEY key;
    T* ptr = new T;
    ar(cereal::make_map_item(key, *ptr));
    auto [it, inserted] = map.emplace(std::move(key), std::move(ptr));
    assert(inserted);
  }
}

} // namespace ankerl::unordered_dense::detail