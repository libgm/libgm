#ifndef LIBGM_TEST_EIGEN_HELPERS_HPP
#define LIBGM_TEST_EIGEN_HELPERS_HPP

#include <libgm/math/eigen/dynamic.hpp>

using libgm::dynamic_vector;
using libgm::dynamic_matrix;

inline dynamic_vector<double>
vec1(double v0) {
  return (dynamic_vector<double>(1) << v0).finished();
}

inline dynamic_vector<double>
vec2(double v0, double v1) {
  return (dynamic_vector<double>(2) << v0, v1).finished();
}

inline dynamic_vector<double>
vec3(double v0, double v1, double v2) {
  return (dynamic_vector<double>(3) << v0, v1, v2).finished();
}

inline dynamic_vector<double>
vec4(double v0, double v1, double v2, double v3) {
  return (dynamic_vector<double>(4) << v0, v1, v2, v3).finished();
}

inline dynamic_vector<double>
vec5(double v0, double v1, double v2, double v3, double v4) {
  return (dynamic_vector<double>(5) << v0, v1, v2, v3, v4).finished();
}

inline dynamic_matrix<double>
mat11(double v00) {
  return (dynamic_matrix<double>(1, 1) << v00).finished();
}

inline dynamic_matrix<double>
mat12(double v00, double v01) {
  return (dynamic_matrix<double>(1, 2) << v00, v01).finished();
}

inline dynamic_matrix<double>
mat21(double v00, double v10) {
  return (dynamic_matrix<double>(2, 1) << v00, v10).finished();
}

inline dynamic_matrix<double>
mat22(double v00, double v01, double v10, double v11) {
  return (dynamic_matrix<double>(2, 2)
          << v00, v01, v10, v11).finished();
}

inline dynamic_matrix<double>
mat23(double v00, double v01, double v02,
      double v10, double v11, double v12) {
  return (dynamic_matrix<double>(2, 3)
          << v00, v01, v02, v10, v11, v12).finished();
}

inline dynamic_matrix<double>
mat33(double v00, double v01, double v02,
      double v10, double v11, double v12,
      double v20, double v21, double v22) {
  return (dynamic_matrix<double>(3, 3)
          << v00, v01, v02, v10, v11, v12, v20, v21, v22).finished();
}

#endif
