#ifndef LIBGM_COUNTING_OUTPUT_ITERATOR_HPP
#define LIBGM_COUNTING_OUTPUT_ITERATOR_HPP

#include <iterator>

#include <libgm/global.hpp>

namespace libgm {

  /**
   * An output iterator that counts how many times a value has been stored.
   * Similarly to insert iterators in STL, operator* returns itself, so that
   * operator= can increase the counter.
   * \ingroup iterator
   */
  class counting_output_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {

  public:
    //! Constructor. 
    counting_output_iterator() : counter_() {}

    //! Increment the counter.
    template <typename T>
    counting_output_iterator& operator=(const T&) {
      ++counter_;
      return *this;
    }

    //! Dereferences to en empty target object that can absorb any assignment.
    counting_output_iterator& operator*() {
      return *this;
    }

    //! Returns *this with no side-effects.
    counting_output_iterator& operator++() {
      return *this;
    }

    //! Returns *this with no side-effects.
    counting_output_iterator& operator++(int) {
      return *this;
    }

    //! Returns the number of positions that have been assigned.
    size_t count() const {
      return counter_;
    }

  private:
    //! The counter.
    size_t counter_;

  }; // class counting_output_iterator

} // namespace libgm

#endif
