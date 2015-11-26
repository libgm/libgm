#ifndef LIBGM_STATIC_RANGE_HPP
#define LIBGM_STATIC_RANGE_HPP

namespace libgm {

  /**
   * A structure that encapsulates a list of indices.
   * This is motivated by
   * http://stackoverflow.com/questions/14261183/how-to-make-generic-computations-over-heterogeneous-argument-packs-of-a-variadic
   */
  template <std::size_t... Is>
  struct index_list { };

  namespace detail {
    template <std::size_t Start, std::size_t Stop, std::size_t... Is>
    struct static_range_impl {
      typedef typename static_range_impl<Start, Stop-1, Stop-1, Is...>::type type;
    };

    template <std::size_t Start, std::size_t... Is>
    struct static_range_impl<Start, Start, Is...> {
      typedef index_list<Is...> type;
    };
  } // namspace detail


  /**
   * An alias for index_list that consists of indices in the half-open range
   * {Start, Start+1, ..., Stop-1}.
   */
  template <std::size_t Start, std::size_t Stop>
  using static_range = typename detail::static_range_impl<Start, Stop>::type;

} // namespace libgm

#endif
