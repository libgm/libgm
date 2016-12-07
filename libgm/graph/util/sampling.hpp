#ifndef LIBGM_GRAPH_SAMPLING_HPP
#define LIBGM_GRAPH_SAMPLING_HPP

#include <random>

namespace libgm {

  template <typename Key, typename T, typename Generator>
  Key sample_key(const std::unordered_map<Key, T>& map, Generator& rng) {
    assert(!map.empty());
    std::uniform_int_distribution<std::size_t> ub(0, map.bucket_count() - 1);
    while (true) {
      std::size_t bucket = ub(rng);
      std::size_t bsize = map.bucket_size(bucket);
      if (bsize > 0) {
        std::uniform_int_distribution<std::size_t> ui(0, bsize - 1);
        return std::next(map.begin(bucket), ui(rng))->first;
      }
    }
  }

} // namespace libgm

#endif
