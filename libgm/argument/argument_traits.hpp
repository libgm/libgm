#ifndef LIBGM_ARGUMENT_TRAITS_HPP
#define LIBGM_ARGUMENT_TRAITS_HPP

#include <libgm/graph/vertex_traits.hpp>

#include <functional>
#include <iosfwd>
#include <type_traits>

namespace libgm {

  // Tags
  //============================================================================

  /**
   * A tag that denotes a univariate argument.
   * See the UnivariateArgument concept.
   */
  struct univariate_tag { };

  /**
   * A tag that denotes a multivariate argument.
   * See the MultivariateArgument concept.
   */
  struct multivariate_tag { };

  /**
   * A tag that denotes a discrete argument category.
   * See the DiscreteArgument concept.
   */
  struct discrete_tag { };

  /**
   * A tag that denotes a continuous argument category.
   * See the ContinuousArgument concept.
   */
  struct continuous_tag { };

  /**
   * A tag that denotes a mixed argument category.
   * See the MixedArgument concept.
   */
  struct mixed_tag : discrete_tag, continuous_tag { };

  // Argument traits primary template
  //============================================================================

  /**
   * A class that specifies the traits for an argument. By default, this class
   * simply forwards the calls to the argument's member functions, except for
   * print(), which gets forwarded to vertex_traits<Arg>::print. The clients
   * of the library can specialize this class to provide a different behavior.
   *
   * \see fixed_discrete_traits, fixed_continuous_traits
   */
  template <typename Arg>
  struct argument_traits {

    /**
     * The arity of the argument. This tag type specifies whether the
     * argument is univariate or multivariate.
     *
     * \see univariate_tag, multivarite_tag
     */
    typedef typename Arg::argument_arity argument_arity;

    /**
     * The category of the argument. This tag type specifies whether the
     * argument is discrete, continuous, or mixed.
     *
     * \see discrete_tag, continuous_tag, mixed_tag
     */
    typedef typename Arg::argument_category argument_category;

    /**
     * The type representing the argument internally, stripped off its index.
     * Allows the argument to participate in conversions.
     */
    typedef typename Arg::descriptor descriptor;

    /**
     * The type representing the argument index if the argument is indexed
     * or indexable and void if it is not.
     */
    typedef typename Arg::index_type index_type;

    /**
     * The type representing the argument instantiated for a specific index
     * if the argument is indexable and void if it is not.
     */
    typedef typename Arg::instance_type instance_type;

    /**
     * The hash function used when this argument is used as a key in
     * unordered associative containers. By default, this is the
     * hasher specified by vertex_traits, which in turn defaults
     * to std::hash.
     */
    typedef typename vertex_traits<Arg>::hasher hasher;

    /**
     * Prints the argument to an output stream.
     */
    static void print(std::ostream& out, Arg arg) {
      vertex_traits<Arg>::print(out, arg);
    }

    /**
     * Returns true if two arguments are compatible. In general, two arguments
     * are compatible if one can be substituted for the other.
     */
    static bool compatible(Arg arg1, Arg arg2) {
      return Arg::compatible(arg1, arg2);
    }

    /**
     * Returns the dimensionality of the argument.
     */
    static std::size_t num_dimensions(Arg arg) {
      return arg.num_dimensions();
    }

    /**
     * Returns the number of values of an argument. For multivariate arguments,
     * this is the total number of assignments to all the components of the
     * argument.
     * This function is only supported for discrete arguments.
     */
    template <bool B =
              std::is_convertible<argument_category, discrete_tag>::value>
    static typename std::enable_if<B, std::size_t>::type
    num_values(Arg arg) {
      return arg.num_values();
    }

    /**
     * Returns the number of values of an argument at a particular position.
     * This function is only supported for discrete multivariate arguments.
     */
    template <bool B =
              std::is_convertible<argument_category, discrete_tag>::value &&
              std::is_same<argument_arity, multivariate_tag>::value>
    static typename std::enable_if<B, std::size_t>::type
    num_values(Arg arg, std::size_t pos) {
      return arg.num_values(pos);
    }

    /**
     * Returns true if the argument is discrete.
     * This function is only supported for mixed arguments.
     */
    template <bool B = std::is_convertible<argument_category, mixed_tag>::value>
    static typename std::enable_if<B, bool>::type discrete(Arg arg) {
      return arg.discrete();
    }

    /**
     * Returns true if the argument is continuous.
     * This function is only supported for mixed arguments.
     */
    template <bool B = std::is_convertible<argument_category, mixed_tag>::value>
    static typename std::enable_if<B, bool>::type continuous(Arg arg) {
      return arg.continuous();
    }

    /**
     * Returns the descriptor of the argument.
     * This function is only supported for arguments with non-void descriptors.
     */
    template <bool B = !std::is_void<descriptor>::value>
    static typename std::enable_if<B, descriptor>::type desc(Arg arg) {
      return arg.desc();
    }

    /**
     * Returns true if this particular argument is indexed.
     * This function is only supported for indexed arguments.
     */
    template <bool B = !std::is_void<index_type>::value &&
                       std::is_void<instance_type>::value>
    static typename std::enable_if<B, bool>::type indexed(Arg arg) {
      return arg.indexed();
    }

    /**
     * Returns the index of the argument.
     * This function is only supported for indexed arguments.
     */
    template <bool B = !std::is_void<index_type>::value &&
                       std::is_void<instance_type>::value>
    static typename std::enable_if<B, index_type>::type index(Arg arg) {
      return arg.index();
    }

  }; // struct argument_traits

  // Derived argument traits
  //============================================================================

  /**
   * Evaluates to std::true_type if Arg is an univariate argument.
   * \see UnivariateArgument
   */
  template <typename Arg>
  struct is_univariate
    : std::is_same<typename argument_traits<Arg>::argument_arity,
                   univariate_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a multivariate argument.
   * \see MultivariateArgument
   */
  template <typename Arg>
  struct is_multivariate
    : std::is_same<typename argument_traits<Arg>::argument_arity,
                   multivariate_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a discrete argument.
   * \see DiscreteArgument
   */
  template <typename Arg>
  struct is_discrete
    : std::is_convertible<typename argument_traits<Arg>::argument_category,
                          discrete_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a continuous argument.
   * \see ContinuousArgument
   */
  template <typename Arg>
  struct is_continuous
    : std::is_convertible<typename argument_traits<Arg>::argument_category,
                          continuous_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a mixed argument.
   * \see MixedArgument
   */
  template <typename Arg>
  struct is_mixed
    : std::is_convertible<typename argument_traits<Arg>::argument_category,
                          mixed_tag> { };

  /**
   * Evaluates to std::true_type if Arg is an indexed argument.
   * \see IndexedArgument
   */
  template <typename Arg>
  struct is_indexed
    : std::integral_constant<
        bool,
        !std::is_void<typename argument_traits<Arg>::index_type>::value &&
        std::is_void<typename argument_traits<Arg>::instance_type>::value
      > { };

  /**
   * Evaluates to std::true_type if Arg is an indexable argument.
   * \see IndexableArgument
   */
  template <typename Arg>
  struct is_indexable
    : std::integral_constant<
        bool,
        !std::is_void<typename argument_traits<Arg>::index_type>::value &&
        !std::is_void<typename argument_traits<Arg>::instance_type>::value
      > { };

  /**
   * Evaluates to std::true_type if Arg1 is convertible to Arg2. One argument
   * is convertible to another if the descriptor of the former is convertible
   * to the descriptor of the latter, and the index_type of the former is
   * convertible to the index_type of the latter.
   */
  template <typename Arg1, typename Arg2>
  struct is_convertible_argument
    : std::integral_constant<
        bool,
        !std::is_void<typename argument_traits<Arg2>::descriptor>::value
        &&
        std::is_convertible<typename argument_traits<Arg1>::descriptor,
                            typename argument_traits<Arg2>::descriptor>::value
        &&
        std::is_convertible<typename argument_traits<Arg1>::index_type,
                            typename argument_traits<Arg2>::index_type>::value
      > { };

  // Statically-sized argument traits
  //============================================================================

  /**
   * A class that implements the traits of a univariate discrete argument with
   * a fixed number of values N. The class uses the hasher and the printer from
   * vertex_traits. In order to use this implementation, specialize the
   * argument_traits for Arg, with this class as the base:
   *
   * template<> struct argument_traits<your_type>
   *   : fixed_discrete_traits<your_type, 5> { };
   *
   * \see argument_traits
   */
  template <typename Arg, std::size_t N>
  struct fixed_discrete_traits {

    //! The arity of the argument.
    typedef univariate_tag argument_arity;

    //! The category of the argument.
    typedef discrete_tag argument_category;

    //! The descriptor of the argument (none).
    typedef void descriptor;

    //! The index type associated with the argument (none).
    typedef void index_type;

    //! The instance of the argument (none).
    typedef void instance_type;

    //! The hash function used on the argument.
    typedef typename vertex_traits<Arg>::hasher hasher;

    //! Prints the argument to an output stream.
    static void print(std::ostream& out, Arg arg) {
      vertex_traits<Arg>::print(out, arg);
    }

    //! Returns true if two arguments are compatible (always true).
    static bool compatible(Arg arg1, Arg arg2) {
      return true;
    }

    //! Returns the dimensionality of the argument (always 1).
    static std::size_t num_dimensions(Arg arg) {
      return 1;
    }

    //! Returns the number of values the argument can take on (fixed to N).
    static std::size_t num_values(Arg arg) {
      return N;
    }

  }; // struct fixed_discrete_traits

  /**
   * A class that implements the traits of a continuous argument with a fixed
   * number of dimensions N. When N == 1, the argument is univariate.
   * With N > 1, the argument is multivariate. The class uses the hasher and
   * the printer from vertex_traits. In order to use this implementation,
   * specialize the argument_traits for Arg, with this class as the base:
   *
   * template<> struct argument_traits<your_type>
   *   : fixed_continuous_traits<your_type, 2> { };
   *
   * \see argument_traits
   */
  template <typename Arg, std::size_t N = 1>
  struct fixed_continuous_traits {

    //! The arity of the argument.
    typedef multivariate_tag argument_arity;

    //! The category of the argument.
    typedef continuous_tag argument_category;

    //! The descriptor of the argument (none).
    typedef void descriptor;

    //! The index type associated with the argument (none).
    typedef void index_type;

    //! The instance of the argument (none).
    typedef void instance_type;

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

  /**
   * A class that implements the traits of an univariate continuous argument.
   */
  template <typename Arg>
  struct fixed_continuous_traits<Arg, 1> {

    //! The arity of the argument.
    typedef univariate_tag argument_arity;

    //! The category of the argument.
    typedef continuous_tag argument_category;

    //! The descriptor of the argument (none).
    typedef void descriptor;

    //! The index type associated with the argument (none).
    typedef void index_type;

    //! The instance of the argument (none).
    typedef void instance_type;

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
      return 1;
    }

  }; // struct fixed_continuous_traits

} // namespace libgm

#endif
