#ifndef LIBGM_CONVERTED_HPP
#define LIBGM_CONVERTED_HPP

#include <libgm/functional/operators.hpp>
#include <libgm/range/iterator_range.hpp>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range.hpp>

namespace libgm {

  //! \ingroup range_adapters
  template <typename T, typename Range>
  iterator_range<
    boost::transform_iterator<
      converter<T>,
      typename boost::range_iterator<const Range>::type> >
  make_converted(const Range& range) {
    return {
      boost::make_transform_iterator(boost::begin(range), converter<T>()),
      boost::make_transform_iterator(boost::end(range), converter<T>())
    };
  }

} // namespace libgm

#endif
