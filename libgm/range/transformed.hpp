#ifndef LIBGM_RANGE_TRANSFORMED_HPP
#define LIBGM_RANGE_TRANSFORMED_HPP

#include <libgm/range/iterator_range.hpp>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range.hpp>

namespace libgm {

  //! \ingroup range_adapters
  template <typename Range, typename F>
  iterator_range<
    boost::transform_iterator<
      F, typename boost::range_iterator<const Range>::type> >
  make_transformed(const Range& range, F f) {
    return { boost::make_transform_iterator(range.begin(), f),
             boost::make_transform_iterator(range.end(), f) };
  }
  
}

#endif
