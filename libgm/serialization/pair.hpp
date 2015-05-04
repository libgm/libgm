#ifndef LIBGM_SERIALIZE_PAIR_HPP
#define LIBGM_SERIALIZE_PAIR_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <utility>

namespace libgm {

  //! Serializes a pair. \relates oarchive
  template <typename T,typename U>
  oarchive& operator<<(oarchive& ar, const std::pair<T, U>& p) {
    ar << p.first;
    ar << p.second;
    return ar;
  }

  //! Deserializes a pair. \relates iarchive
  template <typename T,typename U>
  inline iarchive& operator>>(iarchive& ar, std::pair<T, U>& p){
    ar >> p.first;
    ar >> p.second;
    return ar;
  }

} // namespace libgm

#endif
