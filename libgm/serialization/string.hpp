#ifndef LIBGM_SERIALIZE_STRING_HPP
#define LIBGM_SERIALIZE_STRING_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <cstring>
#include <string>

namespace libgm {

  //! Serializes a C string. \relates oarchive
  inline oarchive& operator<<(oarchive& ar, const char* s) {
    std::size_t length = strlen(s);
    ar << length;
    ar.serialize_buf(s, length);
    return ar;
  }

  //! Serializes an STL string. \relates oarchive
  inline oarchive& operator<<(oarchive& ar, const std::string& s) {
    std::size_t length = s.length();
    ar << length;
    ar.serialize_buf(s.data(), length);
    return ar;
  }

  //! Deserializes a freshly allocated C string. \relates iarchive
  inline iarchive& operator>>(iarchive& ar, char*& s) {
    std::size_t length;
    ar >> length;
    s = new char[length + 1];
    ar.deserialize_buf(s, length);
    s[length] = 0;
    return ar;
  }

  //! Deserializes an STL string. \relates iarchive
  inline iarchive& operator>>(iarchive& ar, std::string& s) {
    std::size_t length;
    ar >> length;
    s.resize(length);
    ar.deserialize_buf(&s[0], length);
    return ar;
  }

} // namespace libgm

#endif
