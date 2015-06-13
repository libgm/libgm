#ifndef LIBGM_SERIALIZE_EIGEN_HPP
#define LIBGM_SERIALIZE_EIGEN_HPP

#include <libgm/math/eigen/real.hpp>
#include <libgm/serialization/iarchive.hpp>
#include <libgm/serialization/oarchive.hpp>

namespace libgm {

  //! Serializes a dynamic Eigen vector. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const real_vector<T>& vec) {
    ar.serialize_int(vec.rows());
    ar.serialize_buf(vec.data(), vec.size() * sizeof(T));
    return ar;
  }

  //! Serializes a dynamic Eigen matrix. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const real_matrix<T>& mat) {
    ar.serialize_int(mat.rows());
    ar.serialize_int(mat.cols());
    ar.serialize_buf(mat.data(), mat.size() * sizeof(T));
    return ar;
  }

  //! Deserializes a dynamic Eigen vector. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, real_vector<T>& vec) {
    std::size_t rows = ar.deserialize_int();
    vec.resize(rows);
    ar.deserialize_buf(vec.data(), vec.size() * sizeof(T));
    return ar;
  }

  //! Deserializes a dynamic Eigen matrix. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, real_matrix<T>& mat) {
    std::size_t rows = ar.deserialize_int();
    std::size_t cols = ar.deserialize_int();
    mat.resize(rows, cols);
    ar.deserialize_buf(mat.data(), mat.size() * sizeof(T));
    return ar;
  }

  //! Serializes a 1D dynamic Eigen array. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar,
                       const Eigen::Array<T, Eigen::Dynamic, 1>& a) {
    ar.serialize_int(a.rows());
    ar.serialize_buf(a.data(), a.size() * sizeof(T));
    return ar;
  }

  //! Serializes a 2D dynamic Eigen array. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar,
                       const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& a) {
    ar.serialize_int(a.rows());
    ar.serialize_int(a.cols());
    ar.serialize_buf(a.data(), a.size() * sizeof(T));
    return ar;
  }

  //! Deserializes a 1D dynamic Eigen array. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar,
                       Eigen::Array<T, Eigen::Dynamic, 1>& a) {
    std::size_t rows = ar.deserialize_int();
    a.resize(rows);
    ar.deserialize_buf(a.data(), a.size() * sizeof(T));
    return ar;
  }

  //! Serializes a 2D dynamic Eigen array. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar,
                       Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& a) {
    std::size_t rows = ar.deserialize_int();
    std::size_t cols = ar.deserialize_int();
    a.resize(rows, cols);
    ar.deserialize_buf(a.data(), a.size() * sizeof(T));
    return ar;
  }


} // namespace libgm

#endif
