#ifndef LIBGM_SERIALIZE_MAP_HPP
#define LIBGM_SERIALIZE_MAP_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>
#include <libgm/serialization/pair.hpp>

#include <iterator>
#include <map>

namespace libgm {

  //! Serializes a map. \relates oarchive
  template <typename T, typename U>
  oarchive& operator<<(oarchive& ar, const std::map<T, U>& map) {
    ar.serialize_range(map.begin(), map.end(), map.size());
    return ar;
  }

  //! Deserializes a map. \relates iarchive
  template <typename T, typename U>
  iarchive& operator>>(iarchive& ar, std::map<T, U>& map) {
    map.clear();
    ar.deserialize_range<std::pair<T, U>>(std::inserter(map, map.end()));
    return ar;
  }

} // namespace libgm

#endif
