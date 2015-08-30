#ifndef LIBGM_FUNCTIONAL_UTILITY_HPP
#define LIBGM_FUNCTIONAL_UTILITY_HPP

#include <utility>

namespace libgm {

  //! Returns the first value of a pair.
  struct pair_first {
    template <typename T, typename U>
    const T& operator()(const std::pair<T, U>& value) const {
      return value.first;
    }
  };

  //! Returns the second value of a pair.
  struct pair_second {
    template <typename T, typename U>
    const U& operator()(const std::pair<T, U>& value) const {
      return value.second;
    }
  };

  //! Returns true if the first contained is smaller than the second one.
  struct size_less {
    template <typename Container>
    bool operator()(const Container& x, const Container& y) const {
      return x.size() < y.size();
    }
  };

  //! Returns true if the first contained is larger than the second one.
  struct size_greater {
    template <typename Container>
    bool operator()(const Container& x, const Container& y) const {
      return x.size() > y.size();
    }
  };

  //! Invokes swap on its arguments.
  struct invoke_swap {
    template <typename T>
    void operator()(T& x, T& y) const {
      using std::swap;
      swap(x, y);
    }
  };

  //! Copies or moves the given value into into the specified buffer.
  struct placement_new {
    void* buf;
    explicit placement_new(void* buf) : buf(buf) { }
    template <typename T>
    void operator()(T&& value) const {
      new (buf) typename std::decay<T>::type(std::forward<T>(value));
    }
  };

  //! Invokes the default constructor on the given object.
  struct invoke_constructor {
    template <typename T>
    void operator()(T& value) const {
      new (&value) T();
    }
  };

  //! Invokes the destructor on the given object.
  struct invoke_destructor {
    template <typename T>
    void operator()(T& value) const {
      value.~T();
    }
  };

  //! A functor that invokes the count function on a container.
  template <typename Container>
  struct count_fn {
    const Container* ptr;
    explicit count_fn(const Container& c) : ptr(&c) { }
    std::size_t operator()(const typename Container::key_type& x) const {
      return ptr->count(x);
    }
  };

  //! Returns a functor that invokes the count function on a container.
  template <typename Container>
  count_fn<Container> count_in(const Container& c) {
    return count_fn<Container>(c);
  }

} // namespace libgm

#endif
