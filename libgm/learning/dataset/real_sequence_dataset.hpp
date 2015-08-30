#ifndef LIBGM_REAL_SEQUENCE_DATASET_HPP
#define LIBGM_REAL_SEQUENCE_DATASET_HPP

#include <libgm/learning/dataset/basic_sequence_dataset.hpp>
#include <libgm/math/eigen/real.hpp>

namespace libgm {

  /**
   * A dense dataset that stores observations for continuous-valued sequences.,
   * in memory. Each sample is an Eigen Matrix with rows corresponding to
   * sequences and column corresponding to time steps.
   * The samples are stored in an std::vector.
   *
   * \tparam Arg
   *         A type that models the ContinuousArgument concept.
   *         This type represents an instance of the sequence at one time.
   * \tparam T
   *         A real type representing the weights.
   * \see Dataset
   */
  template <typename Arg, typename T = double>
  using real_sequence_dataset =
    basic_sequence_dataset<Arg, real_vector<T>, real_matrix<T>, T>;

} // namespace libgm

#endif
