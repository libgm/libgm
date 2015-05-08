#ifndef LIBGM_FINITE_INDEX_HPP
#define LIBGM_FINITE_INDEX_HPP

#include <vector>

namespace libgm {

  /**
   * A type that represents a finite sequence of finite values,
   * each in the set {0, ..., n_k-1}.
   *
   * \ingroup datastructure
   */
  typedef std::vector<std::size_t> finite_index;

} // namespace libgm

#endif
