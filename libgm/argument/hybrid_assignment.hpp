#ifndef LIBGM_HYBRID_ASSIGNMENT_HPP
#define LIBGM_HYBRID_ASSIGNMENT_HPP

#include <libgm/argument/argument_traits.hpp>
#include <libgm/argument/hybrid_domain.hpp>
#include <libgm/argument/real_assignment.hpp>
#include <libgm/argument/uint_assignment.hpp>
#include <libgm/math/eigen/hybrid.hpp>

namespace libgm {

  /**
   * An assignment over a set of finite and vector variables.
   *
   * \tparam Arg a type that satisfies the MixedArgument concept
   */
  template <typename Arg, typename T = double>
  class hybrid_assignment
    : public uint_assignment<Arg>,
      public real_assignment<Arg, T> {
  public:

    /*
    //! The value_type of the underlying finite assignment.
    typedef typename uint_assignment<Arg>::value_type uint_value_type;

    //! The value_type of the underlying vector assignment.
    typedef typename real_assignment<Arg, T>::value_type real_value_type;
    */

    // Constructors and operators
    //==========================================================================

    //! Creates an empty hybrid assignment.
    hybrid_assignment() { }

    //! Constructs an assignment with the given uint component.
    hybrid_assignment(const uint_assignment<Arg>& a)
      : uint_assignment<Arg>(a) { }

    //! Constructs an assignment with the given real component.
    hybrid_assignment(const real_assignment<Arg, T>& a)
      : real_assignment<Arg, T>(a) { }

    //! Constructs an assignment with the given components.
    hybrid_assignment(const uint_assignment<Arg>& ua,
                      const real_assignment<Arg, T>& ra)
      : uint_assignment<Arg>(ua),
        real_assignment<Arg, T>(ra) { }

    /*
    //! Constructs an assignment with the contents of initializer lists.
    hybrid_assignment(std::initializer_list<uint_value_type> uinit,
                      std::initializer_list<real_value_type> rinit)
      : uint_assignment<Arg>(uinit),
        real_assignment<Arg, T>(rinit) { }
    */

    //! Assignment operator.
    hybrid_assignment& operator=(const uint_assignment<Arg>& a) {
      uint() = a;
      real().clear();
      return *this;
    }

    //! Assignment operator.
    hybrid_assignment& operator=(const real_assignment<Arg, T>& a) {
      uint().clear();
      real() = a;
      return *this;
    }

    // Accessors
    //==========================================================================
    //! Returns the integral component of this assignment.
    uint_assignment<Arg>& uint() {
      return *this;
    }

    //! Returns the integral component of this assignment.
    const uint_assignment<Arg>& uint() const {
      return *this;
    }

    //! Returns the real component of this assignment.
    real_assignment<Arg, T>& real() {
      return *this;
    }

    //! Returns the real component of this assignment.
    const real_assignment<Arg, T>& real() const {
      return *this;
    }

    // Container
    //==========================================================================

    //! Returns the total number of elements in this assignment.
    std::size_t size() const {
      return uint().size() + real().size();
    }

    //! Returns true if the assignment is empty.
    std::size_t empty() const {
      return uint().empty() && real().empty();
    }

    //! Swaps the contents of two assignments.
    friend void swap(hybrid_assignment& a, hybrid_assignment& b) {
      using std::swap;
      swap(a.uint(), b.uint());
      swap(a.real(), b.real());
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

    // UnorderedAssociativeContainer
    //==========================================================================

    //! Returns 1 if the assignment contains the given variable.
    std::size_t count(Arg v) const {
      if (argument_traits<Arg>::discrete(v)) {
        return uint().count(v);
      } else if (argument_traits<Arg>::continuous(v)) {
        return real().count(v);
      } else {
        return 0;
      }
    }

    //! Removes a variable from the assignment.
    std::size_t erase(Arg v) {
      if (argument_traits<Arg>::discrete(v)) {
        return uint().erase(v);
      } else if (argument_traits<Arg>::continuous(v)) {
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

    // Assignment
    //==========================================================================
    //! Returns the values in this assignment for a subset of arguments.
    hybrid_vector<T> values(const hybrid_domain<Arg>& args) const {
      return { uint().values(args.discrete()),
               real().values(args.continuous()) };
    }

    //! Inserts the keys drawn form a domain and values from a vector.
    std::size_t insert(const hybrid_domain<Arg>& args,
                       const hybrid_vector<T>& values) {
      return uint().insert(args.discrete(), values.uint()) +
        real().insert(args.continuous(), values.real());
    }

    //! Inserts the keys drawn form a domain and values from a vector.
    std::size_t insert_or_assign(const hybrid_domain<Arg>& args,
                                 const hybrid_vector<T>& values) {
      return uint().insert_or_assign(args.discrete(), values.uint()) +
        real().insert_or_assign(args.continuous(), values.real());
    }

    //! Returns true if args are all present in the given assignment.
    friend bool subset(const hybrid_domain<Arg>& args,
                       const hybrid_assignment& a) {
      return subset(args.discrete(), a.uint()) &&
        subset(args.continuous(), a.real());
    }

    //! Returns true if none of args are present in the given assignment.
    friend bool disjoint(const hybrid_domain<Arg>& args,
                         const hybrid_assignment& a) {
      return disjoint(args.discrete(), a.uint()) &&
        disjoint(args.continuous(), a.real());
    }

    /**
     * Prints a hybrid assignment to an output stream.
     * \relates hybrid_assignment
     */
    friend std::ostream&
    operator<<(std::ostream& out, const hybrid_assignment& a) {
      out << a.uint();
      out << a.real();
      return out;
    }
  }; // class hybrid_assignment

} // namespace libgm

#endif
