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
   * A class that specifies certain elementary traits for an argument.
   * By default, these get aliased to the member types / variables of Arg.
   *
   * \see discrete_traits, continuous_traits
   */
  template <typename Arg>
  struct argument_traits {

    /**
     * The arity of the argument. This tag type specifies whether the
     * argument is univariate or multivariate.
     *
     * \see univariate_tag, multivariate_tag
     */
    using argument_arity = typename Arg::argument_arity;

    /**
     * The category of the argument. This tag type specifies whether the
     * argument is discrete, continuous, or mixed.
     *
     * \see discrete_tag, continuous_tag, mixed_tag
     */
    using argument_category = typename Arg::argument_category;

    /**
     * Specifies whether the argument is indexable. An indexable argument
     * supports operator()(Index index), returning an object of type
     * indexed<Arg, Index>.
     */
    static const bool is_indexable = Arg::is_indexable;

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


  // Basic statically-typed argument traits
  //============================================================================

  /**
   * A class that implements the traits of a discrete argument that is either
   * univariate or multivariate depending on the template argument Scalar.
   * The last template argument specifies whether the argument is indexable.
   *
   * Example use:
   *
   * template<> struct argument_traits<your_type>
   *   : discrete_traits<your_type> { };
   *
   * \see argument_traits
   */
  template <typename Arg, bool Scalar = true, bool Indexable = false>
  struct discrete_traits {

    //! The arity of the argument.
    using argument_arity =
      typedef std::conditional_t<Scalar, univariate_tag, multivariate_tag>;

    //! The category of the argument.
    using argument_category = discrete_tag;

    //! Whether or not Arg supports operator()(index).
    static const bool is_indexable = Indexable;

  }; // struct discrete_traits

  /**
   * A class that implements the traits of a continuous argument that is either
   * univariate or multivariate depending on the template argument Scalar.
   * The last template argument specifies whether the argument is indexable.
   *
   * Example use:
   *
   * template<> struct argument_traits<your_type>
   *   : continuous_traits<your_type> { };
   *
   * \see argument_traits
   */
  template <typename Arg, bool Scalar = true, bool Indexable = false>
  struct continuous_traits {

    //! The arity of the argument.
    using argument_arity =
      typedef std::conditional_t<Scalar, univariate_tag, multivariate_tag>;

    //! The category of the argument.
    using argument_category = continuous_tag;

    //! Whether or not Arg supports operator()(index).
    static const bool is_indexable = Indexable;

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
   * Returns the total number of values of a discrete argument.
   * Delegates to arg.size().
   */
  LIBGM_ENABLE_IF_D(is_discrete<Arg>::value, typename Arg)
  inline std::size_t argument_size(Arg arg) {
    return arg.size();
  }

  /**
   * Returns the number of values for a single index of a discrete argument.
   * For univariate arguments, delegates to argument_size(arg);
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
   * Returns true if the argument is continuous.
   */
  template <typename Arg>
  inline bool argument_continuous(Arg arg) {
    return argument_count(arg, argument_category_t<Arg>());
  }

} // namespace libgm

#endif
