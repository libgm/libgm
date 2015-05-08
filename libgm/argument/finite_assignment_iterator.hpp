#ifndef LIBGM_FINITE_ASSIGNMENT_ITERATOR_HPP
#define LIBGM_FINITE_ASSIGNMENT_ITERATOR_HPP

#include <libgm/argument/basic_domain.hpp>
#include <libgm/argument/finite_assignment.hpp>
#include <libgm/range/iterator_range.hpp>

#include <iterator>

namespace libgm {

  /**
   * An iterator that iterates through all possible assignments
   * to the given finite domain, wrapping over when finished.
   * The order of variables in the vector dictates the order of the
   * assignments, with the first variable being the most significant digit.
   *
   * Note that this class should not be used in performance-critical code
   * For example, when computing statistics of a discrete factor, it is
   * more appropriate to perform an operation on with the underlying table.
   * In this manner, the variable-table dimension conversion is performed
   * only once.
   *
   * \ingroup base_types
   */
  template <typename Var = variable>
  class finite_assignment_iterator
    : public std::iterator<std::forward_iterator_tag,
                           const finite_assignment<Var> > {
  public:
    typedef basic_domain<Var> domain_type;
    typedef finite_assignment<Var> assignment_type;

    //! Default constructor. Creates the "end" iterator.
    finite_assignment_iterator()
      : done_(true) { }

    //! Constructs an iterator pointing to the all-0 assignment for the domain.
    explicit finite_assignment_iterator(const domain_type& vars)
      : vars_(vars), done_(false) {
      for (Var v : vars) {
        a_.emplace(v, 0);
      }
    }

    //! Prefix increment.
    finite_assignment_iterator& operator++() {
      for (Var v : vars_) {
        std::size_t value = a_[v] + 1;
        if (value >= v.size()) {
          a_[v] = 0;
        } else {
          a_[v] = value;
          return *this;
        }
      }
      done_ = true;
      return *this;
    }

    //! Postfix increment (inefficient and should be avoided).
    finite_assignment_iterator operator++(int) {
      finite_assignment_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    //! Returns a const reference to the current assignment.
    const assignment_type& operator*() const {
      return a_;
    }

    //! Returns a const pointer to the current assignment.
    const assignment_type* operator->() const {
      return &a_;
    }

    //! Returns truth if the two assignments are the same.
    bool operator==(const finite_assignment_iterator& it) const {
      return done_ ? it.done_ : !it.done_ && a_ == it.a_;
    }

    //! Returns truth if the two assignments are different.
    bool operator!=(const finite_assignment_iterator& it) const {
      return !(*this == it);
    }

    // Private data members
    //==========================================================================
  private:
    //! The ordered vector of variables to make assignments over.
    domain_type vars_;

    //! The current assignment.
    assignment_type a_;

    //! A flag indicating whether the index has wrapped around.
    bool done_;

  }; // class finite_assignment_iterator

  /**
   * Returns a range over all assignments to variables in the domain.
   */
  template <typename Var>
  iterator_range<finite_assignment_iterator<Var> >
  finite_assignments(const basic_domain<Var>& vars) {
    return { finite_assignment_iterator<Var>(vars),
             finite_assignment_iterator<Var>() };
  }

} // namespace libgm

#endif
