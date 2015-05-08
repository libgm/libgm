#ifndef LIBGM_SERIALIZE_EIGEN_HPP
#define LIBGM_SERIALIZE_EIGEN_HPP

#include <libgm/math/eigen/dynamic.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

namespace libgm {

  //! Serializes a dynamic Eigen vector. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const dynamic_vector<T>& vec) {
    ar.serialize_int(vec.rows());
    ar.serialize_buf(vec.data(), vec.size() * sizeof(T));
    return ar;
  }

  //! Serializes a dynamic Eigen matrix. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const dynamic_matrix<T>& mat) {
    ar.serialize_int(mat.rows());
    ar.serialize_int(mat.cols());
    ar.serialize_buf(mat.data(), mat.size() * sizeof(T));
    return ar;
  }

  //! Deserializes a dynamic Eigen vector. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, dynamic_vector<T>& vec) {
    vec.resize(ar.deserialize_int());
    ar.deserialize_buf(vec.data(), vec.size() * sizeof(T));
    return ar;
  }

  //! Serializes a dynamic Eigen matrix. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, dynamic_matrix<T>& mat) {
    std::size_t rows = ar.deserialize_int();
    std::size_t cols = ar.deserialize_int();
    mat.resize(rows, cols);
    ar.deserialize_buf(mat.data(), mat.size() * sizeof(T));
    return ar;
  }

} // namespace libgm

#endif
