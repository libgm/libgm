#ifndef LIBGM_ASSIGNMENT_HPP
#define LIBGM_ASSIGNMENT_HPP

#include <libgm/argument/assignment/hybrid_assignment.hpp>
#include <libgm/argument/assignment/real_assignment.hpp>
#include <libgm/argument/assignment/uint_assignment.hpp>

namespace libgm {

  namespace detail {

    template <typename Arg,
              typename RealType,
              typename Category = argument_category_t<Arg> >
    struct assignment_selector;

    template <typename Arg, typename RealType>
    struct assignment_selector<Arg, RealType, discrete_tag> {
      using type = uint_assignment<Arg>;
    };

    template <typename Arg, typename RealType>
    struct assignment_selector<Arg, RealType, continuous_tag> {
      using type = real_assignment<Arg, RealType>;
    };

    template <typename Arg, typename RealType>
    struct assignment_selector<Arg, RealType, mixed_tag> {
      using type = hybrid_assignment<Arg, RealType>;
    };

  } // namespace detail

  /**
   * The main alias for all assignments.
   */
  template <typename Arg, typename RealType = double>
  using assignment = typename detail::assignment_selector<Arg, RealType>::type;

} // namespace libgm

#endif
