#ifndef LIBGM_REAL_DATASET_HPP
#define LIBGM_REAL_DATASET_HPP

#include <libgm/math/eigen/dense.hpp>
#include <libgm/learning/dataset/basic_dataset.hpp>

namespace libgm {

  /**
   * A dense dataset that stores observations for continuous arguments
   * in memory. The observations are stored in the column-major format.
   *
   * \tparam Arg a type that models the ContinuousArgument concept
   * \tparam T a real type representing the elements and weights
   *
   * \see basic_dataset
   */
  template <typename Arg, typename T = double>
  using real_dataset = basic_dataset<Arg, dense_vector<T>, T>;

} // namespace libgm

#endif
