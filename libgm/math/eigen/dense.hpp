#pragma once

#include <Eigen/Core>

#include <cereal/cereal.hpp>

namespace libgm {

template <typename T = double>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T = double>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace libgm

namespace Eigen {

template <typename ARCHIVE, typename DERIVED>
void save(ARCHIVE& ar, const PlainObjectBase<DERIVED>& object) {
  if constexpr (DERIVED::RowsAtCompileTime == Eigen::Dynamic &&
                DERIVED::ColsAtCompileTime == Eigen::Dynamic) {
    ar(object.rows(), object.cols());
  }
  ar(cereal::make_size_tag(object.size()));
  for (auto it = object.data(), end = it + object.size(); it != end; ++it) {
    ar(*it);
  }
}

template <typename ARCHIVE, typename DERIVED>
void load(ARCHIVE& ar, PlainObjectBase<DERIVED>& object) {
  if constexpr (DERIVED::RowsAtCompileTime == Eigen::Dynamic &&
                DERIVED::ColsAtCompileTime == Eigen::Dynamic) {
    Index rows, cols;
    ar(rows, cols);
    object.resize(rows, cols);
  }

  cereal::size_type size;
  ar(cereal::make_size_tag(size));
  if constexpr (DERIVED::IsVectorAtCompileTime) {
    object.resize(size);
  } else {
    assert(object.size() == size);
  }

  for (auto it = object.data(), end = it + size; it != end; ++it) {
    ar(*it);
  }
}

} // namespace Eigen
