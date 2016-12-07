#ifndef LIBGM_UINT_ASSIGNMENT_ITERATOR_HPP
#define LIBGM_UINT_ASSIGNMENT_ITERATOR_HPP

#include <libgm/argument/traits.hpp>
#include <libgm/argument/assignment/uint_assignment.hpp>
#include <libgm/argument/domain.hpp>
#include <libgm/range/iterator_range.hpp>

#include <iterator>

namespace libgm {

  /**
   * An iterator that iterates through all possible assignments to the given
   * domain. The order of arguments in the domain dictates the order of the
   * assignments, with the first argument being the least significant digit.
   *
   * Note that this class should not be used in performance-critical code
   * For example, when computing statistics of a discrete factor, it is
   * more appropriate to perform an operation on with the underlying table.
   * In this manner, the argument-table dimension conversion is performed
   * only once.
   *
   * \tparam Arg a type that models the DiscreteArgument concept
   * \tparam Arity the arity of Arg (as specified by its argument_traits)
   *
   * \ingroup base_types
   */
  template <typename Arg, typename Arity = argument_arity_t<Arg> >
  class uint_assignment_iterator;

  /**
   * Specialization of uint_assignment_iterator for univariate arguments.
   */
  template <typename Arg>
  class uint_assignment_iterator<Arg, univariate_tag>
    : public std::iterator<std::forward_iterator_tag,
                           const uint_assignment<Arg> > {
    static_assert(is_discrete<Arg>::value,
                  "Arg must be a discrete argument type");

  public:
    //! Default constructor. Creates the "end" iterator.
    uint_assignment_iterator()
      : done_(true) { }

    //! Constructs an iterator pointing to the all-0 assignment for the domain.
    explicit uint_assignment_iterator(const domain<Arg>& args)
      : args_(args), done_(false) {
      for (Arg arg : args) {
        a_.emplace(arg, 0);
      }
    }

    //! Prefix increment. Advances the iterator.
    uint_assignment_iterator& operator++() {
      for (Arg arg : args_) {
        std::size_t& value = a_[arg];
        ++value;
        if (value >= argument_traits<Arg>::num_values(arg)) {
          value = 0;
        } else {
          return *this;
        }
      }
      done_ = true;
      return *this;
    }

    //! Postfix increment (inefficient and should be avoided).
    uint_assignment_iterator operator++(int) {
      uint_assignment_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    //! Returns a const reference to the current assignment.
    const uint_assignment<Arg>& operator*() const {
      return a_;
    }

    //! Returns a const pointer to the current assignment.
    const uint_assignment<Arg>* operator->() const {
      return &a_;
    }

    //! Returns truth if the two assignments are the same.
    bool operator==(const uint_assignment_iterator& it) const {
      return done_ ? it.done_ : !it.done_ && a_ == it.a_;
    }

    //! Returns truth if the two assignments are different.
    bool operator!=(const uint_assignment_iterator& it) const {
      return !(*this == it);
    }

    // Private data members
    //==========================================================================
  private:
    //! The ordered vector of arguments to make assignments over.
    domain<Arg> args_;

    //! The current assignment.
    uint_assignment<Arg> a_;

    //! A flag indicating whether the index has wrapped around.
    bool done_;

  }; // class uint_assignment_iterator<Arg, univariate_tag>


  /**
   * Specialization of uint_assignment_iterator for multivariate arguments.
   */
  template <typename Arg>
  class uint_assignment_iterator<Arg, multivariate_tag>
    : public std::iterator<std::forward_iterator_tag,
                           const uint_assignment<Arg> > {
    static_assert(is_discrete<Arg>::value,
                  "Arg must be a discrete argument type");

  public:
    //! Default constructor. Creates the "end" iterator.
    uint_assignment_iterator()
      : done_(true) { }

    //! Constructs an iterator pointing to the all-0 assignment for the domain.
    explicit uint_assignment_iterator(const domain<Arg>& args)
      : args_(args), done_(false) {
      for (Arg arg : args) {
        std::size_t n = argument_traits<Arg>::num_dimensions(arg);
        a_.emplace(arg, uint_vector(n, 0));
      }
    }

    //! Prefix increment. Advances the iterator.
    uint_assignment_iterator& operator++() {
      for (Arg arg : args_) {
        uint_vector& values = a_[arg];
        for (std::size_t i = 0; i < values.size(); ++i) {
          ++values[i];
          if (values[i] >= argument_traits<Arg>::num_values(arg, i)) {
            values[i] = 0;
          } else {
            return *this;
          }
        }
      }
      done_ = true;
      return *this;
    }

    //! Postfix increment (inefficient and should be avoided).
    uint_assignment_iterator operator++(int) {
      uint_assignment_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    //! Returns a const reference to the current assignment.
    const uint_assignment<Arg>& operator*() const {
      return a_;
    }

    //! Returns a const pointer to the current assignment.
    const uint_assignment<Arg>* operator->() const {
      return &a_;
    }

    //! Returns truth if the two assignments are the same.
    bool operator==(const uint_assignment_iterator& it) const {
      return done_ ? it.done_ : !it.done_ && a_ == it.a_;
    }

    //! Returns truth if the two assignments are different.
    bool operator!=(const uint_assignment_iterator& it) const {
      return !(*this == it);
    }

    // Private data members
    //==========================================================================
  private:
    //! The ordered vector of arguments to make assignments over.
    domain<Arg> args_;

    //! The current assignment.
    uint_assignment<Arg> a_;

    //! A flag indicating whether the index has wrapped around.
    bool done_;

  }; // class uint_assignment_iterator<Arg, univariate_tag>


  /**
   * Returns a range over all assignments to arguments in the domain.
   */
  template <typename Arg>
  iterator_range<uint_assignment_iterator<Arg> >
  uint_assignments(const domain<Arg>& args) {
    return { uint_assignment_iterator<Arg>(args),
             uint_assignment_iterator<Arg>() };
  }

} // namespace libgm

#endif
