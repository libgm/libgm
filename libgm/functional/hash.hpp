#ifndef LIBGM_FUNCTIONAL_HASH_HPP
#define LIBGM_FUNCTIONAL_HASH_HPP

#include <boost/functional/hash.hpp>

#include <functional>
#include <utility>

namespace libgm {

  using boost::hash_combine;
  using boost::hash_range;

  template <typename T, typename U>
  struct pair_hash {
    typedef std::pair<T, U> argument_type;
    typedef std::size_t     result_type;
    std::size_t operator()(const argument_type& pair) const {
      std::size_t seed = 0;
      hash_combine(seed, pair.first);
      hash_combine(seed, pair.second);
      return seed;
    }
  }; // struct pair_hash

} // namespace libgm

#endif
