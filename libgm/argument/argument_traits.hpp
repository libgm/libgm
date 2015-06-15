#ifndef LIBGM_ARGUMENT_TRAITS_HPP
#define LIBGM_ARGUMENT_TRAITS_HPP

#include <libgm/graph/vertex_traits.hpp>

#include <functional>
#include <iosfwd>

namespace libgm {

  // Tags
  //============================================================================

  /**
   * A tag that denotes a discrete argument category.
   * See the DiscreteArgument concept.
   */
  struct discrete_argument_tag { };

  /**
   * A tag that denotes a continuous argument category.
   * See the ContinuousArgument concept.
   */
  struct continuous_argument_tag { };

  /**
   * A tag that denotes a mixed argument category.
   * See the MixedArgument concept.
   */
  struct mixed_argument_tag
    : discrete_argument_tag, continuous_argument_tag { };


  // Defaults for argument traits
  //============================================================================

  /**
   * This class provides some meaningful defaults for arguments traits
   * Specifically, all the functions get forwarded to member functions of
   * the class, print() uses operator<< for ostream, hasher uses std::hash,
   * and argument category is determined from the class's own argument_category.
   *
   * The default template does not define any members; the actual implementation
   * is in the specializations for each tag.
   */
  template <typename Arg, typename Tag>
  struct default_argument_traits { };


  /**
   * Defaults for discrete arguments.
   */
  template <typename Arg>
  struct default_argument_traits<Arg, discrete_argument_tag> {

    //! The category of the argument.
    typedef discrete_argument_tag argument_category;

    //! The hash function used on the argument.
    typedef std::hash<Arg> hasher;

    //! Prints the argument to an output stream.
    static void print(std::ostream& out, Arg arg) {
      out << arg;
    }

    //! Returns true if two arguments are compatible.
    static bool compatible(Arg arg1, Arg arg2) {
      return Arg::compatible(arg1, arg2);
    }

    //! Returns the number of values the argument can take on.
    static std::size_t num_values(Arg arg) {
      return arg.num_values();
    }

  }; // struct default_argument_traits<Arg, discrete_argument_tag>


  /**
   * Defaults for continuous arguments.
   */
  template <typename Arg>
  struct default_argument_traits<Arg, continuous_argument_tag> {

    //! The category of the argument.
    typedef continuous_argument_tag argument_category;

    //! The hash function used on the argument.
    typedef std::hash<Arg> hasher;

    //! Prints the argument to an output stream.
    static void print(std::ostream& out, Arg arg) {
      out << arg;
    }

    //! Returns true if two arguments are compatible.
    static bool compatible(Arg arg1, Arg arg2) {
      return Arg::compatible(arg1, arg2);
    }

    //! Returns the number of dimensions of the argument.
    static std::size_t num_dimensions(Arg arg) {
      return arg.num_dimensions();
    }

  }; // struct default_argument_traits<Arg, continuous_argument_tag>


  /**
   * Defaults for mixed arguments.
   */
  template <typename Arg>
  struct default_argument_traits<Arg, mixed_argument_tag> {

    //! The category of the argument.
    typedef mixed_argument_tag argument_category;

    //! The hash function used on the argument.
    typedef std::hash<Arg> hasher;

    //! Prints the argument to an output stream.
    static void print(std::ostream& out, Arg arg) {
      out << arg;
    }

    //! Returns true if two arguments are compatible.
    static bool compatible(Arg arg1, Arg arg2) {
      return Arg::compatible(arg1, arg2);
    }

    //! Returns the number of values the argument can take on.
    static std::size_t num_values(Arg arg) {
      return arg.num_values();
    }

    //! Returns the number of dimensions of the argument.
    static std::size_t num_dimensions(Arg arg) {
      return arg.num_dimensions();
    }

    //! Returns true if the mixed argument is discrete.
    static bool is_discrete(Arg arg) {
      return arg.is_discrete();
    }

    //! Returns true if the mixed argument is continuous.
    static bool is_continuous(Arg arg) {
      return arg.is_continuous();
    }

  }; // struct default_argument_traits<Arg, mixed_argument_tag>


  /**
   * A class that provides the traits for indexed arguments.
   */
  template <typename Arg, typename Index>
  struct indexed_argument_traits {
    //! The type that represents the index.
    typedef Index argument_index;

    //! Returns the index associated with the arguments of this type.
    static Index index(Arg arg) {
      return arg.index();
    }

    //! Returns true if this particular argument is indexed.
    static bool is_indexed(Arg arg) {
      return arg.is_indexed;
    }
  }; // struct indexed_argument_traits


  /**
   * A class that provides the traits for non-indexed arguments.
   */
  template <typename Arg>
  struct indexed_argument_traits<Arg, void> {
    // This class does not have an index.
    typedef void argument_index;
  };


  /**
   * A class that describes all the traits of an argument.
   */
  template <typename Arg>
  struct argument_traits
    : default_argument_traits<Arg, typename Arg::argument_category>,
      indexed_argument_traits<Arg, typename Arg::argument_index> { };


  // Statically-sized argument traits
  //============================================================================

  /**
   * A class that represents a discrete argument with fixed number of values N.
   * The class inherits the hasher and the printer from vertex_traits.
   */
  template <typename Arg, std::size_t N>
  struct fixed_discrete_traits {

    //! The category of the argument.
    typedef discrete_argument_tag argument_category;

    //! The index associated with the argument (none).
    typedef void argument_index;

    //! The hash function used on the argument.
    typedef typename vertex_traits<Arg>::hasher hasher;

    //! Prints the argument to an output stream using operator<<(ostream&)
    static void print(std::ostream& out, Arg arg) {
      vertex_traits<Arg>::print(out, arg);
    }

    //! Returns true if two arguments are compatible (always true).
    static bool compatible(Arg arg1, Arg arg2) {
      return true;
    }

    //! Returns the number of values the argument can take on (fixed to N).
    static std::size_t num_values(Arg arg) {
      return N;
    }

  }; // struct fixed_discrete_traits


  /**
   * A class that represents a continuous argument with a fixed number
   * of dimensions N. The class inherits the hasher and the printer
   * from vertex_traits.
   */
  template <typename Arg, std::size_t N>
  struct fixed_continuous_traits {

    //! The category of the argument.
    typedef continuous_argument_tag argument_category;

    //! The index associated with the argument (none).
    typedef void argument_index;

    //! The hash function used on the argument.
    typedef typename vertex_traits<Arg>::hasher hasher;

    //! Prints the argument to an output stream.
    static void print(std::ostream& out, Arg arg) {
      vertex_traits<Arg>::print(out, arg);
    }

    //! Returns true if two arguments are compatible.
    static bool compatible(Arg arg1, Arg arg2) {
      return true;
    }

    //! Returns the number of dimensions of the argument.
    static std::size_t num_dimensions(Arg arg) {
      return N;
    }

  }; // struct fixed_continuous_traits

} // namespace libgm

#endif
