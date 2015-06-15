#ifndef LIBGM_HYBRID_ASSIGNMENT_HPP
#define LIBGM_HYBRID_ASSIGNMENT_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/argument/real_assignment.hpp>

namespace libgm {

  /**
   * An assignment over a set of finite and vector variables.
   *
   * \tparam Var a type that satisfies the MixedArgument concept
   */
  template <typename T = double, typename Var = variable>
  class hybrid_assignment
    : public uint_assignment<Var>,
      public real_assignment<T, Var> {
  public:

    //! The traits associated with the variable.
    typedef argument_traits<Var> arg_traits;

    //! The value_type of the underlying finite assignment.
    typedef typename uint_assignment<Var>::value_type uint_value_type;

    //! The value_type of the underlying vector assignment.
    typedef typename real_assignment<T, Var>::value_type real_value_type;

    // Constructors and operators
    //==========================================================================

    //! Creates an empty hybrid assignment.
    hybrid_assignment() { }

    //! Constructs an assignment with the given uint component.
    hybrid_assignment(const uint_assignment<Var>& a)
      : uint_assignment<Var>(a) { }

    //! Constructs an assignment with the given real component.
    hybrid_assignment(const real_assignment<T, Var>& a)
      : real_assignment<T, Var>(a) { }

    //! Constructs an assignment with the given components.
    hybrid_assignment(const uint_assignment<Var>& ua,
                      const real_assignment<T, Var>& ra)
      : uint_assignment<Var>(ua),
        real_assignment<T, Var>(ra) { }

    //! Constructs an assignment with the contents of an initializer list.
    hybrid_assignment(std::initializer_list<uint_value_type> uinit)
      : uint_assignment<Var>(uinit) { }

    //! Constructs an assignment with the contents of an initializer list.
    hybrid_assignment(std::initializer_list<real_value_type> rinit)
      : real_assignment<T, Var>(rinit) { }

    //! Constructs an assignment with the contents of initializer lists.
    hybrid_assignment(std::initializer_list<uint_value_type> uinit,
                      std::initializer_list<real_value_type> rinit)
      : uint_assignment<Var>(uinit),
        real_assignment<T, Var>(rinit) { }

    //! Assignment operator.
    hybrid_assignment& operator=(const uint_assignment<Var>& a) {
      uint() = a;
      real().clear();
      return *this;
    }

    //! Assignment operator.
    hybrid_assignment& operator=(const real_assignment<T, Var>& a) {
      uint().clear();
      real() = a;
      return *this;
    }

    //! Swaps the contents of two assignments.
    friend void swap(hybrid_assignment& a, hybrid_assignment& b) {
      using std::swap;
      swap(a.uint(), b.uint());
      swap(a.real(), b.real());
    }

    // Accessors
    //==========================================================================
    //! Returns the integral component of this assignment.
    uint_assignment<Var>& uint() {
      return *this;
    }

    //! Returns the integral component of this assignment.
    const uint_assignment<Var>& uint() const {
      return *this;
    }

    //! Returns the real component of this assignment.
    real_assignment<T, Var>& real() {
      return *this;
    }

    //! Returns the real component of this assignment.
    const real_assignment<T, Var>& real() const {
      return *this;
    }

    //! Returns the total number of elements in this assignment.
    std::size_t size() const {
      return uint().size() + real().size();
    }

    //! Returns true if the assignment is empty.
    std::size_t empty() const {
      return uint().empty() && real().empty();
    }

    //! Returns 1 if the assignment contains the given variable.
    std::size_t count(Var v) const {
      if (arg_traits::is_discrete(v)) {
        return uint().count(v);
      } else if (arg_traits::is_continuous(v)) {
        return real().count(v);
      } else {
        return 0;
      }
    }

    /**
     * Returns true if two assignments have the same integral and real
     * components.
     */
    friend bool
    operator==(const hybrid_assignment& a, const hybrid_assignment& b) {
      return a.uint() == b.uint() && a.real() == b.real();
    }

    /**
     * Returns true if two assignments do not have the same integral and real
     * components.
     */
    friend bool
    operator!=(const hybrid_assignment& a, const hybrid_assignment& b) {
      return !(a == b);
    }

    // Mutations
    //==========================================================================

    //! Removes a variable from the assignment.
    std::size_t erase(Var v) {
      if (arg_traits::is_discrete(v)) {
        return uint().erase(v);
      } else if (arg_traits::is_continuous(v)) {
        return real().erase(v);
      } else {
        return 0;
      }
    }

    //! Removes all values from the assignment.
    void clear() {
      uint().clear();
      real().clear();
    }

  }; // class hybrid_assignment

  /**
   * Prints a hybrid assignment to an output stream.
   * \relates hybrid_assignment
   */
  template <typename T, typename Var>
  std::ostream&
  operator<<(std::ostream& out, const hybrid_assignment<T, Var>& a) {
    out << a.uint();
    out << a.real();
    return out;
  }

} // namespace libgm

#endif
