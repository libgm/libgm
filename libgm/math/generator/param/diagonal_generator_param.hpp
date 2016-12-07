#ifndef LIBGM_DIAGONAL_GENERATOR_PARAM_HPP
#define LIBGM_DIAGONAL_GENERATOR_PARAM_HPP

#include <iosfwd>

namespace libgm {

  /**
   * The parameters of a diagonal generator.
   */
  template <typename RealType>
  struct diagonal_generator_param {
    RealType lower;
    RealType upper;
    RealType base;

    param_type(RealType lower = real_type(0),
               RealType upper = real_type(1),
               RealType base = real_type(1))
      : lower(lower), upper(upper), base(base) {
      check();
    }

    void check() const {
      assert(lower <= upper);
    }

    friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
      out << p.lower << " " << p.upper << " " << p.base;
      return out;
    }
  };

} // namespace libgm

#endif
