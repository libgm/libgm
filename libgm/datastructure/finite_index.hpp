#ifndef LIBGM_FINITE_INDEX_HPP
#define LIBGM_FINITE_INDEX_HPP

namespace libgm {

  /**
   * A type that represents a finite sequence of finite values,
   * each in the set {0, ..., n_k-1}.
   *
   * \ingroup datastructure
   */
  typedef std::vector<size_t> finite_index;

} // namespace libgm

#endif
