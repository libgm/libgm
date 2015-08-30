#ifndef LIBGM_UINT_DATASET_HPP
#define LIBGM_UINT_DATASET_HPP

#include <libgm/datastructure/uint_vector.hpp>
#include <libgm/learning/dataset/basic_dataset.hpp>

namespace libgm {

  /**
   * A dense dataset that stores observations for discrete arguments
   * in memory. The observations are stored in the column-major format.
   *
   * \tparam Arg a type that models the DiscreteArgument concept
   * \tparam T a real type representing the weights
   *
   * \see basic_dataset
   */
  template <typename Arg, typename T = double>
  using uint_dataset = basic_dataset<Arg, uint_vector, T>;

} // namespace libgm

#endif
