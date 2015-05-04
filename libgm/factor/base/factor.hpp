#ifndef LIBGM_FACTOR_HPP
#define LIBGM_FACTOR_HPP

#include <libgm/global.hpp>

namespace libgm {

  /**
   * The base class of all factors. This class does not perform any 
   * functionality, but declares a virtual destructor, so that factors
   * can be allocated on the heap and cast if needed.
   *
   * For the functions provided by all factors, see the Factor concept.
   *
   * \ingroup factor_types
   */
  class factor {
  protected:
    //! The default constructor with no arguments
    factor() { }

  public:
    virtual ~factor() { }

  }; // class factor

} // namespace libgm

#endif
