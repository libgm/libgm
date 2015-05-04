#ifndef LIBGM_PROPERTY_MAP_HPP
#define LIBGM_PROPERTY_MAP_HPP

#include <boost/property_map/property_map.hpp>

#include <type_traits>

namespace libgm {

  /**
   * A readable property map based on a unary function.
   * \see Boost.PropertyMap
   */
  template <typename Key, typename F>
  class functor_property_map {
  public:
    typedef boost::readable_property_map_tag      category;
    typedef Key                                   key_type;
    typedef typename std::result_of<F(Key)>::type reference;
    typedef typename std::remove_const<
      typename std::remove_reference<reference>::type>::type value_type;

    //! Constructs the property map from a functor.
    functor_property_map(F f)
      : f_(f) { }

    //! Evaluates the function for the given key.
    reference operator[](const Key& key) const {
      return f_(key);
    }

    //! Evaluates the function for the given key.
    friend reference get(const functor_property_map& map, const Key& key) {
      return map.f_(key);
    }

  private:
    F f_;

  }; // class functor_property_map

  /**
   * A convenience function that constructs a readable property map based
   * on a functor.
   *
   * \tparam Key a required argument, must be specified explicitly.
   * \relates functor_property_map
   */
  template <typename Key, typename F>
  functor_property_map<Key, F> make_functor_property_map(const F& f) {
    return functor_property_map<Key, F>(f);
  }

} // namespace libgm

#endif 
