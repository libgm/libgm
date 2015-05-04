#ifndef LIBGM_MAP_KEY_ITERATOR_HPP
#define LIBGM_MAP_KEY_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * This iterator is used to iterate over the key in an associative container.
   * \ingroup iterator
   */
  template <class Map> 
  class map_key_iterator : 
    public std::iterator<typename Map::const_iterator::iterator_category,
                         typename Map::key_type, 
                         typename Map::difference_type> {
  public:
    typedef const typename Map::key_type& reference;

  private:
    typename Map::const_iterator it;
    template <typename It>
    friend bool operator<(const map_key_iterator<It>& it1, 
                          const map_key_iterator<It>& it2);
    template <typename It>
    friend int operator-(const map_key_iterator<It>& it1, 
                         const map_key_iterator<It>& it2);

  public:
    map_key_iterator() : it() { }

    explicit map_key_iterator(typename Map::const_iterator it) : it(it) { } 
    
    reference operator*() const { 
      return it->first;
    }

    map_key_iterator& operator++() {
      ++it;
      return *this;
    }
    
    map_key_iterator operator++(int) { 
      return map_key_iterator(it++); 
    }
    
    bool operator==(const map_key_iterator& other) const {
      return it == other.it;
    }

    bool operator!=(const map_key_iterator& other) const {
      return it != other.it;
    }

    // the following operations are only supported if the iterator
    // has a random access category
    map_key_iterator& operator+=(int difference) {
      it += difference;
      return *this;
    }

    bool operator<(const map_key_iterator& other) {
      return it < other.it;
    }

    int operator-(const map_key_iterator& other) {
      return it - other.it;
    }

  }; // class map_key_iterator

} // namespace libgm

#endif 
