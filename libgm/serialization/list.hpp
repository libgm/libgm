#ifndef LIBGM_SERIALIZE_LIST_HPP
#define LIBGM_SERIALIZE_LIST_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iterator>
#include <list>

namespace libgm {

  //! Serializes a list. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const std::list<T>& list) {
    ar.serialize_range(list.begin(), list.end(), list.size());
    return ar;
  }

  //! Deserializes a list. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, std::list<T>& list) {
    list.clear();
    ar.deserialize_range<T>(std::back_inserter(list));
    return ar;
  }

} // namespace libgm

#endif

