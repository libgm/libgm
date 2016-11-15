#ifndef LIBGM_ARGUMENT_TRAITS_HPP
#define LIBGM_ARGUMENT_TRAITS_HPP

#include <libgm/enable_if.hpp>
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
  struct mixed_tag { };


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
     * \see univariate_tag, multivariate_tag
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

  }; // struct argument_traits


  // Derived argument traits
  //============================================================================

  /**
   * The arity tag of an argument.
   * \relates argument_traits
   */
  template <typename Arg>
  using argument_arity_t = typename argument_traits<Arg>::argument_arity;

  /**
   * The category tag on an argument.
   * \relates argument_traits
   */
  template <typename Arg>
  using argument_category_t = typename argument_traits<Arg>::argument_category;

  /**
   * The descriptor of an argument.
   * \relates argument_traits
   */
  template <typename Arg>
  using argument_descriptor_t = typename argument_traits<Arg>::decriptor;

  /**
   * The hasher of an argument.
   * \relates argument_traits
   */
  template <typename Arg>
  using argument_hasher_t = typename argument_traits<Arg>::hasher;

  /**
   * Evaluates to std::true_type if Arg is an univariate argument.
   * \see UnivariateArgument
   */
  template <typename Arg>
  struct is_univariate
    : std::is_same<argument_arity_t<Arg>, univariate_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a multivariate argument.
   * \see MultivariateArgument
   */
  template <typename Arg>
  struct is_multivariate
    : std::is_same<argument_arity_t<Arg>, multivariate_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a discrete argument.
   * \see DiscreteArgument
   */
  template <typename Arg>
  struct is_discrete
    : std::is_same<argument_category_t<Arg>, discrete_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a continuous argument.
   * \see ContinuousArgument
   */
  template <typename Arg>
  struct is_continuous
    : std::is_same<argument_category_t<Arg>, continuous_tag> { };

  /**
   * Evaluates to std::true_type if Arg is a mixed argument.
   * \see MixedArgument
   */
  template <typename Arg>
  struct is_mixed
    : std::is_same<argument_category_t<Arg>, mixed_tag> { };

  /**
   * Returns true if the Arg has a non-void descriptor.
   */
  template <typename Arg>
  struct has_descriptor
    : negation<std::is_void<argument_descriptor_t<Arg> > > { };

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


  // Basic statically-typed argument traits
  //============================================================================

  /**
   * A class that implements the traits of a discrete argument that is either
   * univariate or multivariate depending on the template argument Scalar.
   *
   * Example use:
   *
   * template<> struct argument_traits<your_type>
   *   : discrete_traits<your_type> { };
   *
   * \see argument_traits
   */
  template <typename Arg, bool Scalar = true>
  struct discrete_traits {

    //! The arity of the argument.
    using argument_arity =
      typedef std::conditional_t<Scalar, univariate_tag, multivariate_tag>;

    //! The category of the argument.
    using argument_category = discrete_tag;

    //! The descriptor of the argument (none).
    using descriptor = void;

    //! The index type associated with the argument (none).
    using index_type = void;

    //! The instance of the argument (none).
    using instance_type = void;

    //! The hash function used on the argument.
    typedef typename vertex_traits<Arg>::hasher hasher;

  }; // struct discrete_traits

  /**
   * A class that implements the traits of a continuous argument that is either
   * univariate or multivariate depending on the template argument Scalar.
   *
   * Example use:
   *
   * template<> struct argument_traits<your_type>
   *   : continuous_traits<your_type> { };
   *
   * \see argument_traits
   */
  template <typename Arg, bool Scalar = true>
  struct continuous_traits {

    //! The arity of the argument.
    using argument_arity =
      typedef std::conditional_t<Scalar, univariate_tag, multivariate_tag>;

    //! The category of the argument.
    using argument_category = continuous_tag;

    //! The descriptor of the argument (none).
    using descriptor = void;

    //! The index type associated with the argument (none).
    using index_type = void;

    //! The instance of the argument (none).
    using instance_type = void;

    //! The hash function used on the argument.
    typedef typename vertex_traits<Arg>::hasher hasher;

  }; // struct continuous_traits


  // Argument properties
  //============================================================================

  namespace detail {

    template <typename Arg>
    inline std::size_t argument_arity(Arg arg, univariate_tag) {
      return 1;
    }

    template <typename Arg>
    inline std::size_t argument_arity(Arg arg, multivariate_tag) {
      return arg.arity();
    }

    template <typename Arg>
    inline std::size_t argument_size(Arg arg, std::size_t i, univariate_tag) {
      assert(i == 0);
      return argument_size(arg);
    }

    template <typename Arg>
    inline std::size_t argument_size(Arg arg, std::size_t i, multivariate_tag) {
      assert(i < argument_arity(arg));
      return arg.size(i);
    }

    template <typename Arg>
    inline bool argument_discrete(Arg /* arg */, discrete_tag) {
      return true;
    }

    template <typename Arg>
    inline bool argument_discrete(Arg /* arg */, continuous_tag) {
      return false;
    }

    template <typename Arg>
    inline bool argument_discrete(Arg arg, mixed_tag) {
      return arg.discrete();
    }

    template <typename Arg>
    inline bool argument_continuous(Arg /* arg */, discrete_tag) {
      return false;
    }

    template <typename Arg>
    inline bool argument_continuous(Arg /* arg */, continuous_tag) {
      return true;
    }

    template <typename Arg>
    inline bool argument_continuous(Arg arg, mixed_tag) {
      return arg.continuous();
    }

  } // namespace detail

  /**
   * Returns the arity (number of dimensions) of the argument.
   * For univariate arguments, this is guaranteed to be 1;
   * for multivariate arguments, this by default delegated to arg.arity().
   */
  template <typename Arg>
  inline std::size_t argument_arity(Arg arg) {
    return argument_arity(arg, argument_arity_t<Arg>());
  }

  /**
   * Returns the number of values of a univariate argument.
   * Delegates to arg.size().
   */
  LIBGM_ENABLE_IF_D(is_discrete<Arg>::value && is_univariate<Arg>::value,
                    typename Arg)
  inline std::size_t argument_size(Arg arg) {
    return arg.size();
  }

  /**
   * Returns the number of values for a single index of an argument.
   * For univariate arguments, delegates to argument_size();
   * for multivariate argument, delegates to arg.size(i).
   */
  LIBGM_ENABLE_IF_D(is_discrete<Arg>::value, typename Arg)
  inline std::size_t argument_size(Arg arg, std::size_t i) {
    return argument_size(arg, i, argument_arity_t<Arg>());
  }

  /**
   * Prints the argument to an output stream.
   * This delegates to operator<<.
   */
  template <typename Arg>
  inline void print_argument(std::ostream& out, Arg arg) {
    out << arg;
  }

  /**
   * Returns true if the argument is discrete.
   * For discrete or continuous arguments, this returns true or false,
   * respectively. For mixed arguments, this delegates to arg.discrete().
   */
  template <typename Arg>
  inline bool argument_discrete(Arg arg) {
    return argument_discrete(arg, argument_category_t<Arg>());
  }

  /**
   * Returns true if the argumetn is continuous.
   */
  template <typename Arg>
  inline bool argument_continuous(Arg arg) {
    return argument_count(arg, argument_category_t<Arg>());
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


} // namespace libgm

#endif
