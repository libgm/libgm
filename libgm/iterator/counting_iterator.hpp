#ifndef LIBGM_COUNTING_ITERATOR_HPP
#define LIBGM_COUNTING_ITERATOR_HPP

#include <iterator>

namespace libgm {

  /**
   * A random access iterator that wraps around an unsigned integer.
   * \ingroup iterator
   */
  class counting_iterator
    : public std::iterator<std::random_access_iterator_tag,
                           std::size_t,
                           std::ptrdiff_t,
                           void,
                           std::size_t> {

  public:
    /* implicit */ counting_iterator(std::size_t value = 0)
      : value_(value) { }

    std::size_t operator*() const {
      return value_;
    }

    counting_iterator& operator++() {
      ++value_;
      return *this;
    }

    counting_iterator operator++(int) {
      return value_ + 1;
    }

    counting_iterator& operator--() {
      --value_;
      return *this;
    }

    counting_iterator operator--(int) {
      return value_ - 1;
    }

    counting_iterator& operator+=(std::ptrdiff_t n) {
      value_ += n;
      return *this;
    }

    counting_iterator& operator-=(std::ptrdiff_t n) {
      value_ -= n;
      return *this;
    }

    friend counting_iterator operator+(counting_iterator it, std::ptrdiff_t n) {
      return it.value_ + n;
    }

    friend counting_iterator operator+(std::ptrdiff_t n, counting_iterator it) {
      return n + it.value_;
    }

    friend counting_iterator operator-(counting_iterator it, std::ptrdiff_t n) {
      return it.value_ - n;
    }

    friend std::ptrdiff_t operator-(counting_iterator a, counting_iterator b) {
      return a.value_ - b.value_;
    }

    std::size_t operator[](std::ptrdiff_t n) const {
      return value_ + n;
    }

    friend bool operator==(counting_iterator a, counting_iterator b) {
      return a.value_ == b.value_;
    }

    friend bool operator!=(counting_iterator a, counting_iterator b) {
      return a.value_ != b.value_;
    }

    friend bool operator<=(counting_iterator a, counting_iterator b) {
      return a.value_ <= b.value_;
    }

    friend bool operator>=(counting_iterator a, counting_iterator b) {
      return a.value_ >= b.value_;
    }

    friend bool operator<(counting_iterator a, counting_iterator b) {
      return a.value_ < b.value_;
    }

    friend bool operator>(counting_iterator a, counting_iterator b) {
      return a.value_ > b.value_;
    }

    friend void swap(counting_iterator& a, counting_iterator& b) {
      std::swap(a.value_, b.value_);
    }

  private:
    std::size_t value_;

  }; // class counting_iterator

} // namespace libgm

#endif
