#ifndef LIBGM_SERIALIZE_BOOST_UNORDERED_SET_HPP
#define LIBGM_SERIALIZE_BOOST_UNORDERED_SET_HPP

#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

#include <iterator>

#include <boost/unordered_set.hpp>

namespace libgm {

  //! Serializes a set. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const boost::unordered_set<T>& set) {
    ar.serialize_range(set.begin(), set.end(), set.size());
    return ar;
  }

  //! Deserializes a set. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& a, boost::unordered_set<T>& set) {
    set.clear();
    ar.deserialize_range<T>(std::inserter(set, set.end()));
    return ar;
  }

} // namespace libgm

#endif
