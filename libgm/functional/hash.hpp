#ifndef LIBGM_FUNCTIONAL_HASH_HPP
#define LIBGM_FUNCTIONAL_HASH_HPP

#include <boost/functional/hash.hpp>

#include <functional>
#include <utility>

namespace libgm {

  using boost::hash_combine;
  using boost::hash_range;

  //! A hash object that delegates to the hash_value() function
  template <typename T>
  struct default_hash {
    typedef T argument_type;
    typedef std::size_t result_type;
    std::size_t operator()(const T& x) const {
      return hash_value(x);
    }
  }; // default_hash

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

  //! Returns the hash value of the given object.
  struct invoke_hash {
    template <typename T>
    std::size_t operator()(const T& value) const {
      using boost::hash_value;
      return hash_value(value);
    }
  };

} // namespace libgm

#endif
