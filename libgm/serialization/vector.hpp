#ifndef LIBGM_SERIALIZE_VECTOR_HPP
#define LIBGM_SERIALIZE_VECTOR_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iterator>
#include <vector>

namespace libgm {

  //! Serializes a vector. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const std::vector<T>& vec){
    ar.serialize_range(vec.begin(), vec.end());
    return ar;
  }

  //! Serializes a vector<bool>. \relates oarchive
  inline oarchive& operator<<(oarchive& ar, const std::vector<bool>& vec) {
    ar << vec.size();
    for (size_t i = 0; i < vec.size(); ++i) {
      ar.serialize_char(vec[i]);
    }
    return ar;
  }

  //! Deserializes a vector. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, std::vector<T>& vec) {
    vec.clear();
    ar.deserialize_range<T>(std::back_inserter(vec));
    return ar;
  }

} // namespace libgm

#endif
