#ifndef LIBGM_RANGE_REVERSED_HPP
#define LIBGM_RANGE_REVERSED_HPP

#include <libgm/range/iterator_range.hpp>

#include <boost/iterator/reverse_iterator.hpp>
#include <boost/range.hpp>

namespace libgm {

  //! \ingroup range_adapters
  template <typename Range>
  iterator_range<
    boost::reverse_iterator<typename boost::range_iterator<const Range>::type> >
  make_reversed(const Range& range) {
    return { boost::make_reverse_iterator(boost::end(range)),
             boost::make_reverse_iterator(boost::begin(range)) };
  }
  
} // namespace libgm

#endif
