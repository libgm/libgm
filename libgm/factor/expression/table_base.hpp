#ifndef LIBGM_TABLE_BASE_HPP
#define LIBGM_TABLE_BASE_HPP

namespace libgm { namespace experimental {

  /**
   * The base class of all table factor expressions.
   * This class must be specialized for each specific factor class.
   */
  template <typename Space, typename RealType, typename Derived>
  class table_base;

} } // namespace libgm::experimental

#endif
