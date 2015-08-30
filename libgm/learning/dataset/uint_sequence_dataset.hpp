#ifndef LIBGM_UINT_SEQUENCE_DATASET_HPP
#define LIBGM_UINT_SEQUENCE_DATASET_HPP

#include <libgm/math/eigen/uint.hpp>
#include <libgm/learning/dataset/basic_sequence_dataset.hpp>

namespace libgm {

  /**
   * A dense dataaset that stores observations for discrete-valued sequences
   * in memory. Each sample is an Eigen Array with rows corresponding to
   * sequences and columns corresponding to time steps.
   * The samples are stored in an std::vector.
   *
   * \tparam Arg
   *         A type that models the DiscreteArgument concept.
   *         This type represents an instance of the sequence at one time.
   * \tparam T
   *         A real type representing the weights.
   * \see Dataset
   */
  template <typename Arg, typename T = double>
  using uint_sequence_dataset =
    basic_sequence_dataset<Arg, uint_vector, uint_matrix, T>;

} // namespace libgm

#endif
