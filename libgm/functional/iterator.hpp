#ifndef LIBGM_FUNCTIONAL_ITERATOR_HPP
#define LIBGM_FUNCTIONAL_ITERATOR_HPP

namespace libgm {

  /**
   * A funciton object that returns the pointer of an object.
   */
  struct address_of {
    template <typename T>
    T* operator()(T& t) const {
      return &t;
    }
  };

  /**
   * A function object that dereferences an iterator.
   */
  struct dereference {
    template <typename It>
    typename std::iterator_traits<It>::reference
    operator()(const It& it) const {
      return *it;
    }
  };

  /**
   * A function object that preincrements an iterator.
   */
  struct preincrement {
    template <typename It>
    It& operator()(It& it) const {
      ++it;
      return it;
    }
  };

} // namespace libgm

#endif
