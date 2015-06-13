#ifndef LIBGM_ARGUMENT_CONCEPTS_HPP
#define LIBGM_ARGUMENT_CONCEPTS_HPP

#include <cstddef>

namespace libgm {

  /**
   * The concept that represents an argument.
   *
   * An argument is DefaultConstructible, CopyAssignable, CopyConstructible,
   * and Destructible, EqualityComparable, LessThanComparable, and Swappable.
   * Many models also need to the argument to specialize std::hash<> and
   * operator<<(std::ostream&, Arg), so that the argument can be used as key
   * in unordered containers and printed to an output stream.
   *
   * \ingroup argument_concepts
   */
  template <typename Arg>
  struct Argument {
    /**
     * Returns true if two arguments are compatible. Two arguments are
     * compatible if one can be substituted for another in the model,
     * while keeping the model well-defined. Typically, this means that
     * the number of values or the dimensionality of the arguments must
     * be the same, but Arg can place additional constraints that are
     * meaningful in the application.
     *
     * Compatibility must satisfy two properties:
     * 1) symmetry: if (x, y) are compatible, so are (y, x);
     * 2) transitivity: if (x, y) are compatible, and (y, z) are
     *    compatible, so are (x, z).
     */
    friend bool compatible(Argument x, Argument y);
  };

  /**
   * The concept that represents a discrete variable taking on a fixed
   * number of values.
   */
  template <typename Arg>
  struct DiscreteArgument : Argument<Arg> {
    //! Returns the number of values a discrete variable can take on.
    friend std::size_t num_values(DiscreteArgument v);
  };

  /**
   * The concept that represents a continuous variable with a fixed
   * number of dimensions.
   */
  template <typename Arg>
  struct ContinuousArgument : Argument<Arg> {
    // Returns the number of dimensions of a continuous variable.
    friend std::size_t num_dimensions(ContinuousArgument v);
  };

  /**
   * The concept that represents an argument that can be either discrete or
   * continuous in nature. A call to num_values is only valid if the
   * argument is discrete (i.e., is_discrete returns true), and a
   * call to num_dimensions is only valid if the argument is continuous
   * (i.e., is_continuous returns true).
   */
  template <typename Arg>
  struct MixedArgument : DiscreteArgument<Arg>, ContinuousArgument<Arg> {
    //! Returns true if this variable represents a discrete variable.
    friend bool is_discrete(MixedArgument v);

    //! Returns true if this variable represents a continuous variable.
    friend bool is_continuous(MixedArgument v);
  };

  /**
   * The concept that represents a variable associated with a process.
   */
  template <typename Var>
  struct ProcessVariable : Argument<Arg> {
    /**
     * Returns a value convertible to the index of the process this
     * variable is associated with. The caller needs to (implicitly)
     * convert this value to the index_type of the process before using it.
     */
    friend auto index(ProcessVariable v);

    /**
     * Returns true if the variable is associated with a process
     * (optional).
     */
    friend bool is_indexed(ProcessVariable v);
  };

  /**
   * The concept that represents a random process.
   */
  template <typename Proc>
  struct Process : Argument<Arg> {
    //! The index type of the variables represented by this process.
    typedef typename Proc::index_type index_type;

    //! The type used to index the variables of this process
    typedef typename Proc::variable_type variable_type;

    //! Returns the variable for the given index
    variable_type operator()(const index_type& index) const;
  };

} // namespace libgm

#endif
