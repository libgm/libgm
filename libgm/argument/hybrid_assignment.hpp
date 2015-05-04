#ifndef LIBGM_HYBRID_ASSIGNMENT_HPP
#define LIBGM_HYBRID_ASSIGNMENT_HPP

#include <libgm/argument/finite_assignment.hpp>
#include <libgm/argument/vector_assignment.hpp>

namespace libgm {

  /**
   * An assignment over a set of finite and vector variables.
   */
  template <typename T = double, typename Var = variable>
  class hybrid_assignment
    : public finite_assignment<Var>,
      public vector_assignment<T, Var> {

  public:

    //! The value_type of the underlying finite assignment.
    typedef typename finite_assignment<Var>::value_type finite_value_type;

    //! The value_type of the underlying vector assignment.
    typedef typename vector_assignment<T, Var>::value_type vector_value_type;

    // Constructors and operators
    //==========================================================================

    //! Creates an empty hybrid assignment.
    hybrid_assignment() { }

    //! Constructs an assignment with the given finite component.
    hybrid_assignment(const finite_assignment<Var>& a)
      : finite_assignment<Var>(a) { }

    //! Constructs an assignment with the given vector component.
    hybrid_assignment(const vector_assignment<T, Var>& a)
      : vector_assignment<T, Var>(a) { }

    //! Constructs an assignment with the given components.
    hybrid_assignment(const finite_assignment<Var>& fa,
                      const vector_assignment<T, Var>& va)
      : finite_assignment<Var>(fa),
        vector_assignment<T, Var>(va) { }

    //! Constructs an assignment with the contents of an initializer list.
    hybrid_assignment(std::initializer_list<finite_value_type> finit)
      : finite_assignment<Var>(finit) { }

    //! Constructs an assignment with the contents fo an initializer list.
    hybrid_assignment(std::initializer_list<vector_value_type> vinit)
      : vector_assignment<T, Var>(vinit) { }

    //! Constructs an assignment with the contents of initializer lists.
    hybrid_assignment(std::initializer_list<finite_value_type> finit,
                      std::initializer_list<vector_value_type> vinit)
      : finite_assignment<Var>(finit),
        vector_assignment<T, Var>(vinit) { }

    //! Assignment operator.
    hybrid_assignment& operator=(const finite_assignment<Var>& a) {
      finite() = a;
      vector().clear();
      return *this;
    }

    //! Assignment operator.
    hybrid_assignment& operator=(const vector_assignment<T, Var>& a) {
      finite().clear();
      vector() = a;
      return *this;
    }

    //! Swaps the contents of two assignments.
    friend void swap(hybrid_assignment& a, hybrid_assignment& b) {
      using std::swap;
      swap(a.finite(), b.finite());
      swap(a.vector(), b.vector());
    }

    // Accessors
    //==========================================================================
    //! Returns the finite component of this assignment.
    finite_assignment<Var>& finite() {
      return *this;
    }

    //! Returns the finite component of this assignment.
    const finite_assignment<Var>& finite() const {
      return *this;
    }

    //! Returns the vector component of this assignment.
    vector_assignment<T, Var>& vector() {
      return *this;
    }

    //! Returns the vector component of this assignment.
    const vector_assignment<T, Var>& vector() const {
      return *this;
    }

    //! Returns the total number of elements in this assignment.
    size_t size() const {
      return finite().size() + vector().size();
    }

    //! Returns true if the assignment is empty.
    size_t empty() const {
      return finite().empty() && vector().empty();
    }

    //! Returns 1 if the assignment contains the given variable.
    bool count(Var v) const {
      if (v.finite()) {
        return finite().count(v);
      } else if (v.vector()) {
        return vector().count(v);
      } else {
        return 0;
      }
    }

    /**
     * Returns true if two assignments have the same finite and vector
     * componets.
     */
    friend bool
    operator==(const hybrid_assignment& a, const hybrid_assignment& b) {
      return a.finite() == b.finite() && a.vector() == b.vector();
    }

    /**
     * Returns true if two assignments do not have the same finite and
     * vector components.
     */
    friend bool
    operator!=(const hybrid_assignment& a, const hybrid_assignment& b) {
      return !(a == b);
    }

    // Mutations
    //==========================================================================

    //! Removes a variable from the assignment.
    size_t erase(variable v) {
      if (v.finite()) {
        return finite().erase(v);
      } else if (v.vector()) {
        return vector().erase(v);
      } else {
        return 0;
      }
    }

    //! Removes all values from the assignment.
    void clear() {
      finite().clear();
      vector().clear();
    }

  }; // class hybrid_assignment

  /**
   * Prints a hybrid assignment to an output stream.
   * \relates hybrid_assignment
   */
  template <typename T, typename Var>
  std::ostream& 
  operator<<(std::ostream& out, const hybrid_assignment<T, Var>& a) {
    out << a.finite();
    out << a.vector();
    return out;
  }

} // end namespace libgm

#endif
